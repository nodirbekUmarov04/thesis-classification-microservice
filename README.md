# Noise Source Classification Microservice

FastAPI microservice for WAV audio classification.

The service accepts a WAV file, splits long audio into 5-second chunks, ignores the last chunk if it is shorter than 2 seconds, predicts a class for every chunk, and returns one final class by majority vote.

## Classes

The model returns one of these classes:

```text
transport
human
alert
building_noise
animals
others
```

## Required Model Files

Before running the service, make sure these files exist in the `artifacts/` directory:

```text
artifacts/
  sound_classifier_best.keras
  sound_classifier_metadata.json
```

If `sound_classifier_best.keras` is missing, the service will try to use:

```text
artifacts/sound_classifier_final.keras
```

The metadata file must match the model preprocessing settings.

If these files are committed to git, Docker will copy them into the image during build.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the service:

```bash
python main.py
```

Or run with Uvicorn directly:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Docker Run

Build the image:

```bash
docker build -t noise-classifier .
```

Run the container:

```bash
docker run --rm -p 8000:8000 noise-classifier
```

If you want to use local model files without rebuilding the image, mount `artifacts/`:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" noise-classifier
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Docker Compose

Run:

```bash
docker compose up --build
```

If your Docker installation uses the legacy Compose command:

```bash
docker-compose up --build
```

The service will be available at:

```text
http://127.0.0.1:8000
```

## API

### GET `/health`

Checks that the service is running and the model is loaded.

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "model": "sound_classifier_best.keras",
  "classes": ["transport", "human", "alert", "building_noise", "animals", "others"]
}
```

### POST `/predict`

Classifies a WAV file.

Request type:

```text
multipart/form-data
```

File field name:

```text
file
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@audio.wav"
```

Optional confidence threshold:

```bash
curl -X POST "http://127.0.0.1:8000/predict?confidence_threshold=0.45" -F "file=@audio.wav"
```

Response:

```json
{
  "label": "transport",
  "confidence": 0.977,
  "chunks_count": 3,
  "duration_seconds": 14.711
}
```

## Response Fields

```text
label              final predicted class
confidence         average confidence for chunks that voted for the final class
chunks_count       number of processed 5-second chunks
duration_seconds   original audio duration
```

## Error Cases

Only `.wav` files are supported.

If the audio is shorter than 2 seconds:

```json
{
  "detail": "Audio is too short. Minimum duration is 2 seconds."
}
```

## Logs

The service logs:

```text
model loading
incoming filename and file size
audio duration
number of kept/skipped chunks
prediction for each chunk
final prediction
```

## Notes for Hosting

For local network access:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

For Docker hosting, expose port `8000`. If the model is committed to git, it is copied into the image. If you want to swap models without rebuilding, mount the `artifacts/` directory.
