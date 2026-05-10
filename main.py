import io
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
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


def load_wav(audio_bytes: bytes) -> np.ndarray:
    try:
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read audio. Please upload a valid WAV file.",
        ) from exc

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    return audio.astype(np.float32)


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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.45, ge=0.0, le=1.0),
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
    duration_seconds = float(len(audio) / SAMPLE_RATE)
    logger.info(
        "Loaded audio: filename=%s duration=%.3fs samples=%s sample_rate=%s",
        filename,
        duration_seconds,
        len(audio),
        SAMPLE_RATE,
    )

    chunks = split_audio(audio)

    if not chunks:
        logger.warning(
            "Rejected short audio: filename=%s duration=%.3fs min_duration=%ss",
            filename,
            duration_seconds,
            MIN_CHUNK_DURATION,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Audio is too short. Minimum duration is {MIN_CHUNK_DURATION} seconds.",
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
        "label": final_prediction["label"],
        "confidence": round(final_prediction["confidence"], 4),
        "chunks_count": len(chunks),
        "duration_seconds": round(duration_seconds, 3),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
