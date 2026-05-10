import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile


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
N_MELS = int(MODEL_METADATA["n_mels"])
MAX_LEN = int(MODEL_METADATA["max_len"])
TRAIN_MEAN = float(MODEL_METADATA["train_mean"])
TRAIN_STD = float(MODEL_METADATA["train_std"])

model = tf.keras.models.load_model(MODEL_PATH)

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

    for start in range(0, len(audio), SAMPLES):
        chunk = audio[start : start + SAMPLES]

        if len(chunk) < SAMPLES:
            chunk = np.pad(chunk, (0, SAMPLES - len(chunk)), mode="constant")

        chunks.append(chunk.astype(np.float32))

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


def predict_chunk(chunk: np.ndarray, confidence_threshold: float) -> dict[str, Any]:
    features = extract_mel_from_chunk(chunk)
    probabilities = model.predict(features[np.newaxis, ...], verbose=0)[0]

    raw_class_id = int(np.argmax(probabilities))
    raw_label = CLASS_NAMES[raw_class_id]
    confidence = float(probabilities[raw_class_id])
    label = raw_label if confidence >= confidence_threshold else "others"

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


def choose_final_label(chunk_predictions: list[dict[str, Any]]) -> str:
    label_counts = Counter(prediction["label"] for prediction in chunk_predictions)
    top_count = max(label_counts.values())
    candidates = {
        label for label, count in label_counts.items()
        if count == top_count
    }

    if len(candidates) == 1:
        return next(iter(candidates))

    confidence_sums = defaultdict(float)

    for prediction in chunk_predictions:
        if prediction["label"] in candidates:
            confidence_sums[prediction["label"]] += prediction["confidence"]

    return max(candidates, key=lambda label: confidence_sums[label])


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

    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )

    audio_bytes = await file.read()
    audio = load_wav(audio_bytes)
    chunks = split_audio(audio)
    chunk_predictions = [
        predict_chunk(chunk, confidence_threshold)
        for chunk in chunks
    ]
    final_label = choose_final_label(chunk_predictions)

    return {
        "label": final_label,
        "chunks_count": len(chunks),
        "duration_seconds": round(float(len(audio) / SAMPLE_RATE), 3),
        "confidence_threshold": confidence_threshold,
        "chunks": [
            {
                "chunk_index": index,
                "start_seconds": index * DURATION,
                "end_seconds": min((index + 1) * DURATION, len(audio) / SAMPLE_RATE),
                **prediction,
            }
            for index, prediction in enumerate(chunk_predictions)
        ],
    }
