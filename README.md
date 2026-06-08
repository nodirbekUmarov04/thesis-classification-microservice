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

By default RabbitMQ integration is disabled in this standalone compose file. Enable it with:

```bash
RABBITMQ_ENABLED=true docker compose up --build
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
  "classes": ["transport", "human", "alert", "building_noise", "animals", "others"],
  "rabbitmq_enabled": false
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

Optional calibration offset:

```bash
curl -X POST "http://127.0.0.1:8000/predict?calibration_offset=96" -F "file=@audio.wav"
```

If `calibration_offset` is not provided, the service uses the default value `96.0`.
You can override it with the environment variable `DEFAULT_CALIBRATION_OFFSET`.
If the provided calibration value is missing or outside the valid range, the service
falls back to the default value.

Response:

```json
{
  "sound_class": "transport",
  "noise_level_dba": 70.0
}
```

## Response Fields

```text
sound_class   final predicted class
noise_level_dba   calibrated loudness estimate
```

Loudness is calculated as:

```text
noise_level_dba = volume_dbfs + calibration_offset
```

## RabbitMQ Integration

The service is compatible with the `noisemap` backend event flow.

It consumes:

```text
exchange: noisemap.events
routing key: recording.created
queue: ml.classification.queue
```

Expected incoming event fields:

```json
{
  "recordingId": "...",
  "userId": "...",
  "audioFileUrl": "/data/audio/.../recording.wav",
  "latitude": 43.238,
  "longitude": 76.945,
  "calibrationOffset": 96.0,
  "recordedAt": "2026-05-10T10:00:00Z"
}
```

It publishes:

```text
exchange: noisemap.events
routing key: classification.completed
```

Published event:

```json
{
  "recordingId": "...",
  "userId": "...",
  "latitude": 43.238,
  "longitude": 76.945,
  "noiseLevelDba": 70.0,
  "noiseClass": "transport",
  "confidenceScore": 0.977,
  "recordedAt": "2026-05-10T10:00:00Z",
  "classifiedAt": "2026-05-10T10:00:05Z"
}
```

The service also declares and binds `recording.classification.result.queue` to `classification.completed`, so `recording-service` can update the recording status.

### RabbitMQ Environment Variables

```text
RABBITMQ_ENABLED=false
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
DEFAULT_CALIBRATION_OFFSET=96.0
AUDIO_STORAGE_MOUNT=/data/audio
```

### Add to noisemap docker-compose.yml

When running inside the `noisemap` backend compose network, add a service like this:

```yaml
  ml-classification-service:
    build: ../thesis-classification-microservice
    ports:
      - "8000:8000"
    environment:
      RABBITMQ_ENABLED: "true"
      RABBITMQ_HOST: rabbitmq
      RABBITMQ_PORT: 5672
      RABBITMQ_USER: guest
      RABBITMQ_PASS: guest
      DEFAULT_CALIBRATION_OFFSET: 96.0
      AUDIO_STORAGE_MOUNT: /data/audio
    volumes:
      - audio_data:/data/audio:ro
    depends_on:
      rabbitmq:
        condition: service_healthy
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
