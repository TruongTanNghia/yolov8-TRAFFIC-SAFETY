# Helmet Guard AI

Local MVP built with FastAPI, SQLite, OpenCV, Ultralytics YOLO tracking, and a Vietnamese web dashboard. It uses the existing `models/v8s_ECA_B.pt` checkpoint. No sample events are seeded; KPI values come from SQLite.

## Run

Python 3.10+ is recommended. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766`. API docs are at `/docs`; health/model state is at `/api/health`. Choose another local port if 8766 is already in use.

## Video input

- **Upload:** MP4, AVI, MOV, MKV, WEBM up to 500 MB. After upload, the browser opens a WebSocket that processes and displays frames with detections and ByteTrack IDs.
- **Webcam:** uses browser `getUserMedia`, requests camera permission, and streams JPEG frames to the backend. Open the UI on localhost; browser camera access requires localhost or HTTPS.
- **RTSP:** enter an `rtsp://` or `rtsps://` URL. OpenCV/FFmpeg uses TCP transport. Credentials are sent to the local backend over the WebSocket; URL-encode reserved characters in credentials. The backend machine must be able to reach the camera and FFmpeg must support its codec. Do not expose the unauthenticated MVP outside a trusted local machine.
- Only one inference stream runs at a time to protect tracker state and GPU memory. Live sources retry after frame loss with bounded exponential backoff.
- Confidence and overlay are adjustable. Use Stop to close the capture and release resources.

## Existing checkpoint

The checkpoint serializes `ultralytics.nn.modules.conv.ECAAttention`, which is absent from the installed Ultralytics package. `model_runtime.py` registers the Efficient Channel Attention layer under that expected module path before loading. The checkpoint then loads on CUDA when available (CPU fallback) and reports these classes: `Motorcyclist`, `Plate`, `Helmet`, `Non_helmet`.

The event heuristic requires a tracked `Non_helmet` box whose center lies within a `Motorcyclist` box for 3 consecutive frames by default (`HELMET_EVENT_CONFIRM_FRAMES` can change it). This is a review candidate heuristic, not proof that the rider is the driver or a legal violation. Track ID is temporary within the stream.

## Evidence and review

On a candidate event the backend saves overview, annotated, and padded crop JPEGs under `artifacts/evidence/`, then stores the event and metadata in SQLite. The Events page shows real rows; the review dialog displays evidence and saves reviewer status/note with an audit record. Evidence is served through event-scoped API routes rather than arbitrary file paths.

## Implemented APIs

- `GET /api/health`, `GET /api/dashboard/stats`
- `POST /api/videos/upload`, `GET /api/jobs`
- `GET /api/events`, `GET /api/events/{id}`
- `GET /api/events/{id}/evidence/{overview|annotated|crop}`
- `PATCH /api/events/{id}/review`, `GET /api/models`
- WebSocket `/api/stream` for upload, webcam, and RTSP inference

## Limits and safety

- There is no independent dataset or benchmark report yet. Precision, recall, mAP, event accuracy, FPS, and latency must be measured on authorized representative data. Blank-frame smoke tests check the stream path, not detection quality.
- Model labels and the overlap/temporal heuristic cannot reliably establish who is driving, handle every passenger/occlusion case, or distinguish every helmet state. A human must review every event.
- Plate boxes may be detected for this model but the MVP does not perform plate OCR or identity matching.
- Review API currently has no authentication/roles. Keep bound to localhost. No retention scheduler, privacy blur, or legal workflow is implemented.
- Process only video sources you are authorized to use. Do not use for face recognition, identity inference, or automatic penalties.

## Files

```text
app.py             FastAPI endpoints, SQLite repository, stream loop
model_runtime.py   ECA compatibility, model loader, YOLO + ByteTrack inference
web/               Vietnamese dashboard and WebSocket client
models/            Supplied checkpoint (not modified)
data/               Runtime SQLite and video uploads
artifacts/evidence/ Runtime event evidence
docs/               Architecture and dataset plan
```
