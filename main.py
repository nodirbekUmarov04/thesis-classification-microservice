from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
import pika
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("noise-classifier")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "sound_classifier_best.keras"
METADATA_PATH = ARTIFACTS_DIR / "sound_classifier_metadata.json"
DEFAULT_CALIBRATION_OFFSET = float(os.getenv("DEFAULT_CALIBRATION_OFFSET", "96.0"))

RABBITMQ_ENABLED = os.getenv("RABBITMQ_ENABLED", "false").lower() == "true"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", os.getenv("RABBIT_USER", "guest"))
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", os.getenv("RABBIT_PASS", "guest"))
RABBITMQ_EXCHANGE = os.getenv("RABBITMQ_EXCHANGE", "noisemap.events")
RABBITMQ_RECORDING_CREATED_KEY = os.getenv("RABBITMQ_RECORDING_CREATED_KEY", "recording.created")
RABBITMQ_CLASSIFICATION_COMPLETED_KEY = os.getenv(
    "RABBITMQ_CLASSIFICATION_COMPLETED_KEY",
    "classification.completed",
)
RABBITMQ_ML_QUEUE = os.getenv("RABBITMQ_ML_QUEUE", "ml.classification.queue")
RABBITMQ_RECORDING_RESULT_QUEUE = os.getenv(
    "RABBITMQ_RECORDING_RESULT_QUEUE",
    "recording.classification.result.queue",
)
AUDIO_STORAGE_MOUNT = os.getenv("AUDIO_STORAGE_MOUNT", "/data/audio")

if not MODEL_PATH.exists():
    MODEL_PATH = ARTIFACTS_DIR / "sound_classifier_final.keras"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file was not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise RuntimeError(f"Model metadata file was not found: {METADATA_PATH}")


with open(METADATA_PATH, "r", encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)


CLASS_NAMES = MODEL_METADATA["class_names"]
SAMPLE_RATE = int(MODEL_METADATA["sample_rate"])
DURATION = int(MODEL_METADATA["duration"])
SAMPLES = int(MODEL_METADATA["samples"])
MIN_CHUNK_DURATION = 2
MIN_CHUNK_SAMPLES = SAMPLE_RATE * MIN_CHUNK_DURATION
N_MELS = int(MODEL_METADATA["n_mels"])
MAX_LEN = int(MODEL_METADATA["max_len"])
TRAIN_MEAN = float(MODEL_METADATA["train_mean"])
TRAIN_STD = float(MODEL_METADATA["train_std"])

logger.info("Loading model from %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded. Classes: %s", ", ".join(CLASS_NAMES))

app = FastAPI(
    title="Noise Source Classification Microservice",
    description="Accepts a WAV file and returns the most likely noise source class.",
    version="1.0.0",
)


@app.on_event("startup")
def start_rabbitmq_worker() -> None:
    if not RABBITMQ_ENABLED:
        logger.info("RabbitMQ worker is disabled. Set RABBITMQ_ENABLED=true to enable it.")
        return

    thread = threading.Thread(
        target=run_rabbitmq_worker,
        name="rabbitmq-worker",
        daemon=True,
    )
    thread.start()
    logger.info("RabbitMQ worker thread launched.")


def decode_wav_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise ValueError("Could not read audio. Please upload a valid WAV file.") from exc

    if audio.size == 0:
        raise ValueError("Uploaded audio is empty.")

    return audio.astype(np.float32)


def load_wav(audio_bytes: bytes) -> np.ndarray:
    try:
        return decode_wav_audio(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def split_audio(audio: np.ndarray) -> list[np.ndarray]:
    chunks = []
    skipped_chunks = 0

    for start in range(0, len(audio), SAMPLES):
        chunk = audio[start : start + SAMPLES]
        original_chunk_duration = len(chunk) / SAMPLE_RATE

        if len(chunk) < MIN_CHUNK_SAMPLES:
            skipped_chunks += 1
            logger.info(
                "Skipping short chunk: start=%.3fs duration=%.3fs min_duration=%ss",
                start / SAMPLE_RATE,
                original_chunk_duration,
                MIN_CHUNK_DURATION,
            )
            continue

        if len(chunk) < SAMPLES:
            chunk = np.pad(chunk, (0, SAMPLES - len(chunk)), mode="constant")

        chunks.append(chunk.astype(np.float32))

    logger.info(
        "Audio split completed: kept_chunks=%s skipped_chunks=%s",
        len(chunks),
        skipped_chunks,
    )
    return chunks


def calculate_volume(audio: np.ndarray) -> dict[str, float | str]:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    dbfs = float(20 * np.log10(max(rms, 1e-12)))

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "dbfs": round(dbfs, 2),
    }


def extract_mel_from_chunk(chunk: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_LEN - mel_db.shape[1])),
            mode="constant",
        )
    else:
        mel_db = mel_db[:, :MAX_LEN]

    mel_db = (mel_db.astype(np.float32) - TRAIN_MEAN) / TRAIN_STD
    return mel_db[..., np.newaxis]


def predict_chunk(
    chunk: np.ndarray,
    confidence_threshold: float,
    chunk_index: int,
) -> dict[str, Any]:
    features = extract_mel_from_chunk(chunk)
    probabilities = model.predict(features[np.newaxis, ...], verbose=0)[0]

    raw_class_id = int(np.argmax(probabilities))
    raw_label = CLASS_NAMES[raw_class_id]
    confidence = float(probabilities[raw_class_id])
    label = raw_label if confidence >= confidence_threshold else "others"

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = ", ".join(
        f"{CLASS_NAMES[index]}={probabilities[index]:.3f}"
        for index in top_indices
    )
    logger.info(
        "Chunk %s prediction: label=%s raw_label=%s confidence=%.4f top3=[%s]",
        chunk_index,
        label,
        raw_label,
        confidence,
        top_predictions,
    )

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


def choose_final_prediction(chunk_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(prediction["label"] for prediction in chunk_predictions)
    top_count = max(label_counts.values())
    candidates = {
        label for label, count in label_counts.items()
        if count == top_count
    }

    if len(candidates) == 1:
        final_label = next(iter(candidates))
    else:
        confidence_sums = defaultdict(float)

        for prediction in chunk_predictions:
            if prediction["label"] in candidates:
                confidence_sums[prediction["label"]] += prediction["confidence"]

        final_label = max(candidates, key=lambda label: confidence_sums[label])

    final_confidences = [
        prediction["confidence"]
        for prediction in chunk_predictions
        if prediction["label"] == final_label
    ]

    return {
        "label": final_label,
        "confidence": float(np.mean(final_confidences)),
    }


def classify_audio(
    audio: np.ndarray,
    filename: str,
    calibration_offset: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    duration_seconds = float(len(audio) / SAMPLE_RATE)
    logger.info(
        "Loaded audio: filename=%s duration=%.3fs samples=%s sample_rate=%s",
        filename,
        duration_seconds,
        len(audio),
        SAMPLE_RATE,
    )

    chunks = split_audio(audio)
    volume = calculate_volume(audio)
    noise_level_dba = volume["dbfs"] + calibration_offset
    logger.info(
        "Audio volume: filename=%s rms=%.6f peak=%.6f dbfs=%.2f calibration_offset=%.2f dba=%.2f",
        filename,
        volume["rms"],
        volume["peak"],
        volume["dbfs"],
        calibration_offset,
        noise_level_dba,
    )

    if not chunks:
        raise ValueError(
            f"Audio is too short. Minimum duration is {MIN_CHUNK_DURATION} seconds."
        )

    chunk_predictions = [
        predict_chunk(chunk, confidence_threshold, index)
        for index, chunk in enumerate(chunks)
    ]
    final_prediction = choose_final_prediction(chunk_predictions)

    logger.info(
        "Final prediction: filename=%s label=%s confidence=%.4f chunks_count=%s",
        filename,
        final_prediction["label"],
        final_prediction["confidence"],
        len(chunks),
    )

    return {
        "sound_class": final_prediction["label"],
        "noise_level_dba": round(noise_level_dba, 2),
        "confidence_score": round(final_prediction["confidence"], 4),
        "duration_seconds": round(duration_seconds, 3),
        "chunks_count": len(chunks),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_audio_bytes_from_url(audio_file_url: str) -> bytes:
    if not audio_file_url:
        raise ValueError("audioFileUrl is empty.")

    if audio_file_url.startswith(("http://", "https://")):
        logger.info("Downloading audio from URL: %s", audio_file_url)
        with urllib.request.urlopen(audio_file_url, timeout=30) as response:
            return response.read()

    if audio_file_url.startswith("file://"):
        audio_file_url = audio_file_url.removeprefix("file://")

    file_path = Path(audio_file_url)

    if not file_path.exists() and audio_file_url.startswith("/data/audio/"):
        relative_path = audio_file_url.removeprefix("/data/audio/").lstrip("/")
        file_path = Path(AUDIO_STORAGE_MOUNT) / relative_path

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file was not found: {audio_file_url}")

    logger.info("Reading audio file from path: %s", file_path)
    return file_path.read_bytes()


def build_classification_completed_event(
    recording_event: dict[str, Any],
    classification_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recordingId": recording_event.get("recordingId"),
        "userId": recording_event.get("userId"),
        "latitude": recording_event.get("latitude"),
        "longitude": recording_event.get("longitude"),
        "noiseLevelDba": classification_result["noise_level_dba"],
        "noiseClass": classification_result["sound_class"],
        "confidenceScore": classification_result["confidence_score"],
        "recordedAt": recording_event.get("recordedAt"),
        "classifiedAt": utc_now_iso(),
    }


def rabbitmq_connection() -> pika.BlockingConnection:
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
    )
    return pika.BlockingConnection(parameters)


def setup_rabbitmq_channel(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    channel.exchange_declare(
        exchange=RABBITMQ_EXCHANGE,
        exchange_type="topic",
        durable=True,
    )
    channel.queue_declare(queue=RABBITMQ_ML_QUEUE, durable=True)
    channel.queue_bind(
        queue=RABBITMQ_ML_QUEUE,
        exchange=RABBITMQ_EXCHANGE,
        routing_key=RABBITMQ_RECORDING_CREATED_KEY,
    )
    channel.queue_declare(queue=RABBITMQ_RECORDING_RESULT_QUEUE, durable=True)
    channel.queue_bind(
        queue=RABBITMQ_RECORDING_RESULT_QUEUE,
        exchange=RABBITMQ_EXCHANGE,
        routing_key=RABBITMQ_CLASSIFICATION_COMPLETED_KEY,
    )
    channel.basic_qos(prefetch_count=1)


def publish_classification_completed(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    event: dict[str, Any],
) -> None:
    body = json.dumps(event, ensure_ascii=False).encode("utf-8")
    properties = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
        headers={
            "__TypeId__": "kz.noisemap.common.event.ClassificationCompletedEvent",
        },
    )
    channel.basic_publish(
        exchange=RABBITMQ_EXCHANGE,
        routing_key=RABBITMQ_CLASSIFICATION_COMPLETED_KEY,
        body=body,
        properties=properties,
    )


def process_recording_created_event(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    recording_event: dict[str, Any],
) -> None:
    recording_id = recording_event.get("recordingId")
    audio_file_url = recording_event.get("audioFileUrl")
    calibration_offset = recording_event.get("calibrationOffset")

    if calibration_offset is None:
        calibration_offset = DEFAULT_CALIBRATION_OFFSET

    logger.info(
        "Processing recording.created: recordingId=%s audioFileUrl=%s calibrationOffset=%.2f",
        recording_id,
        audio_file_url,
        float(calibration_offset),
    )

    audio_bytes = read_audio_bytes_from_url(audio_file_url)
    audio = decode_wav_audio(audio_bytes)
    classification_result = classify_audio(
        audio=audio,
        filename=str(audio_file_url),
        calibration_offset=float(calibration_offset),
        confidence_threshold=0.45,
    )
    completed_event = build_classification_completed_event(
        recording_event,
        classification_result,
    )
    publish_classification_completed(channel, completed_event)
    logger.info(
        "Published classification.completed: recordingId=%s noiseClass=%s noiseLevelDba=%.2f confidence=%.4f",
        completed_event["recordingId"],
        completed_event["noiseClass"],
        completed_event["noiseLevelDba"],
        completed_event["confidenceScore"],
    )


def run_rabbitmq_worker() -> None:
    while True:
        try:
            logger.info(
                "Connecting to RabbitMQ: host=%s port=%s queue=%s",
                RABBITMQ_HOST,
                RABBITMQ_PORT,
                RABBITMQ_ML_QUEUE,
            )
            connection = rabbitmq_connection()
            channel = connection.channel()
            setup_rabbitmq_channel(channel)

            def callback(
                ch: pika.adapters.blocking_connection.BlockingChannel,
                method: pika.spec.Basic.Deliver,
                properties: pika.BasicProperties,
                body: bytes,
            ) -> None:
                try:
                    recording_event = json.loads(body.decode("utf-8"))
                    process_recording_created_event(ch, recording_event)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    logger.exception("Failed to process RabbitMQ message.")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(
                queue=RABBITMQ_ML_QUEUE,
                on_message_callback=callback,
            )
            logger.info("RabbitMQ worker started.")
            channel.start_consuming()
        except Exception:
            logger.exception("RabbitMQ worker stopped unexpectedly. Retrying in 5 seconds.")
            time.sleep(5)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "classes": CLASS_NAMES,
        "rabbitmq_enabled": RABBITMQ_ENABLED,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.45, ge=0.0, le=1.0),
    calibration_offset: float = Query(DEFAULT_CALIBRATION_OFFSET),
) -> dict[str, Any]:
    filename = file.filename or ""
    logger.info("Received prediction request: filename=%s", filename)

    if not filename.lower().endswith(".wav"):
        logger.warning("Rejected file with unsupported extension: filename=%s", filename)
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )

    audio_bytes = await file.read()
    logger.info("Read uploaded file: filename=%s size_bytes=%s", filename, len(audio_bytes))

    audio = load_wav(audio_bytes)
    try:
        result = classify_audio(
            audio=audio,
            filename=filename,
            calibration_offset=calibration_offset,
            confidence_threshold=confidence_threshold,
        )
    except ValueError as exc:
        logger.warning(
            "Rejected audio: filename=%s reason=%s",
            filename,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "sound_class": result["sound_class"],
        "noise_level_dba": result["noise_level_dba"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
