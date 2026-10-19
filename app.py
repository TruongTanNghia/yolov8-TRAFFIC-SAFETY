"""Helmet Guard AI MVP: local dashboard and evidence review API."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "helmet_guard.db"
UPLOADS = ROOT / "data" / "uploads"
MODEL_PATH = Path(os.getenv("HELMET_MODEL", ROOT / "models" / "v8s_ECA_B.pt"))
EVIDENCE = ROOT / "artifacts" / "evidence"
WEB = ROOT / "web"
DB.parent.mkdir(exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)
EVENT_CONFIRM_FRAMES = max(2, int(os.getenv("HELMET_EVENT_CONFIRM_FRAMES", "3")))

from model_runtime import INFERENCE_LOCK, LABELS, device_info, infer_track, load_model, reset_tracker

app = FastAPI(title="Helmet Guard AI", version="0.1.0", description="Local MVP for reviewing helmet safety events")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@contextmanager
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY, created_at TEXT NOT NULL, camera TEXT NOT NULL,
      confidence REAL NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
      evidence TEXT NOT NULL DEFAULT '')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
      status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')""")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@lru_cache(maxsize=1)
def model_status():
    try:
        model = load_model()
        return {"loaded": True, "path": str(MODEL_PATH), "error": None, "device": device_info(), "labels": LABELS}
    except Exception as exc:
        return {"loaded": False, "path": str(MODEL_PATH), "error": f"Không nạp được checkpoint: {type(exc).__name__}: {exc}"}


class Review(BaseModel):
    status: str = Field(pattern="^(confirmed|rejected|needs_review)$")
    note: str = Field(default="", max_length=1000)
    reviewer: str = Field(default="Admin", max_length=100)


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    m = model_status()
    ffmpeg = any("FFMPEG" in line and "YES" in line for line in cv2.getBuildInformation().splitlines())
    return {"status": "ok", "model": m, "database": "sqlite", "video": {"opencv": cv2.__version__, "ffmpeg": ffmpeg, "rtsp_transport": "tcp"}, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/dashboard/stats")
def stats():
    with db() as c:
        rows = c.execute("SELECT status,COUNT(*) n FROM events GROUP BY status").fetchall()
        jobs = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    counts = {r["status"]: r["n"] for r in rows}
    return {"events": sum(counts.values()), "needs_review": counts.get("needs_review", 0), "confirmed": counts.get("confirmed", 0), "rejected": counts.get("rejected", 0), "videos_processed": jobs, "active_cameras": 0}


@app.get("/api/events")
def events(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    if status and status not in {"needs_review", "confirmed", "rejected"}:
        raise HTTPException(400, "Trạng thái không hợp lệ")
    limit = min(max(limit, 1), 100)
    with db() as c:
        sql = "SELECT * FROM events" + (" WHERE status=?" if status else "") + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        rows = c.execute(sql, ([status, limit, offset] if status else [limit, offset])).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/events/{event_id}")
def event_detail(event_id: str):
    with db() as c:
        row = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Không tìm thấy sự kiện")
    result = dict(row)
    try:
        result["evidence"] = json.loads(result["evidence"])
    except (TypeError, json.JSONDecodeError):
        result["evidence"] = {}
    return result


@app.get("/api/events/{event_id}/evidence/{kind}")
def event_evidence(event_id: str, kind: str):
    if kind not in {"overview", "annotated", "crop"}:
        raise HTTPException(404, "Loại ảnh không hợp lệ")
    with db() as c:
        row = c.execute("SELECT evidence FROM events WHERE id=?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Không tìm thấy sự kiện")
    try:
        filename = json.loads(row["evidence"])[kind]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise HTTPException(404, "Ảnh bằng chứng không tồn tại")
    path = (EVIDENCE / Path(filename).name).resolve()
    if path.parent != EVIDENCE.resolve() or not path.is_file():
        raise HTTPException(404, "Ảnh bằng chứng không tồn tại")
    return FileResponse(path, media_type="image/jpeg")


def _find_upload(job_id: str) -> Path:
    with db() as c:
        row = c.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise ValueError("Không tìm thấy job video")
    paths = list(UPLOADS.glob(f"{job_id}.*"))
    if len(paths) != 1:
        raise ValueError("File video của job không tồn tại")
    return paths[0]


def _open_capture(kind: str, source):
    if kind == "rtsp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    elif kind == "webcam" and os.name == "nt":
        cap = None
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            attempt = cv2.VideoCapture(source, backend)
            if attempt.isOpened():
                ok, _ = attempt.read()
                if ok:
                    cap = attempt
                    break
            attempt.release()
        if cap is None:
            raise RuntimeError(f"Không mở được webcam index {source} qua DirectShow/MSMF. Kiểm tra camera, quyền truy cập camera của Windows và đóng ứng dụng khác đang giữ camera.")
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh bằng chứng thành JPEG.")
    path.write_bytes(encoded.tobytes())


def _draw_detections(frame: np.ndarray, result) -> np.ndarray:
    """Draw compact labels with stable sizing instead of Ultralytics' large default plot."""
    image = frame.copy()
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return image
    xyxy = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
    h,w=image.shape[:2]
    scale=max(.42,min(.68,min(h,w)/1000))
    thickness=max(1,round(scale*2))
    colors={0:(238,173,76),1:(214,155,55),2:(78,203,133),3:(70,78,239)}
    names={0:"Rider",1:"Plate",2:"Helmet",3:"No helmet"}
    occupied=[]
    for i in sorted(range(len(xyxy)),key=lambda n:(xyxy[n][1],xyxy[n][0])):
        x1,y1,x2,y2=[int(round(v)) for v in xyxy[i]]
        x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
        if x2<=x1 or y2<=y1: continue
        cid=int(classes[i]);color=colors.get(cid,(210,210,210))
        cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness,cv2.LINE_AA)
        tid=ids[i]
        label=f"{names.get(cid,'Object')} {float(scores[i]):.2f}"+(f" #{int(tid)}" if tid is not None else "")
        (tw,th),base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
        lx=max(0,min(w-tw-10,x1));label_h=th+base+7;step=label_h+2
        candidates=[max(0,y1-label_h)]+list(range(y1,max(y1,y2-label_h+1),step))
        ly=None
        for candidate_y in candidates:
            rect=(lx,candidate_y,min(w-1,lx+tw+10),min(h-1,candidate_y+label_h))
            if all(rect[2]<ax or rect[0]>bx or rect[3]<ay or rect[1]>by for ax,ay,bx,by in occupied):
                ly=candidate_y;break
        if ly is None:continue
        bottom=min(h-1,ly+th+base+7);right=min(w-1,lx+tw+10)
        cv2.rectangle(image,(lx,ly),(right,bottom),color,-1)
        cv2.putText(image,label,(lx+5,min(h-3,ly+th+2)),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
        occupied.append((lx,ly,right,bottom))
    return image


def _save_event(frame: np.ndarray, annotated: np.ndarray, box: list[float], score: float, camera: str, stream_id: str, track_id: int) -> dict:
    event_id = f"EVT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    prefix = f"{event_id}-{uuid.uuid4().hex[:6]}"
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pad_x, pad_y = max(12, (x2-x1)//4), max(12, (y2-y1)//4)
    crop = frame[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
    files = {"overview": f"{prefix}-overview.jpg", "annotated": f"{prefix}-annotated.jpg", "crop": f"{prefix}-crop.jpg"}
    _write_jpeg(EVIDENCE/files["overview"], frame, 92)
    _write_jpeg(EVIDENCE/files["annotated"], annotated, 90)
    if crop.size:
        _write_jpeg(EVIDENCE/files["crop"], crop, 92)
    metadata = {**files, "tracking_id": track_id, "stream_id": stream_id, "model": MODEL_PATH.name, "model_labels": LABELS, "reviewer": None}
    with db() as c:
        c.execute("INSERT INTO events(id,created_at,camera,confidence,status,note,evidence) VALUES(?,?,?,?,?,?,?)", (event_id, datetime.now(timezone.utc).isoformat(), camera, float(score), "needs_review", "", json.dumps(metadata)))
    return {"id": event_id, "camera": camera, "confidence": round(float(score), 4), "status": "needs_review", "tracking_id": track_id}


@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    cap = None
    acquired = False
    job_id = None
    job_finished = False
    try:
        cfg = await asyncio.wait_for(ws.receive_json(), timeout=20)
        kind = cfg.get("type")
        stream_id = str(uuid.uuid4())
        if kind == "upload":
            job_id = str(cfg.get("job_id", ""))
            source = _find_upload(job_id)
            camera = f"VIDEO-{job_id[:8]}"
            source_arg = str(source)
        elif kind == "webcam":
            index = int(cfg.get("index", 0))
            if index < 0 or index > 4:
                raise ValueError("Chỉ số webcam hợp lệ từ 0 đến 4")
            source_arg = index
            camera = f"WEBCAM-{index}"
        elif kind == "browser_webcam":
            source_arg = None
            camera = "WEBCAM-BROWSER"
        elif kind == "rtsp":
            source_arg = str(cfg.get("url", "")).strip()
            parsed = urlparse(source_arg)
            if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
                raise ValueError("URL cần có dạng rtsp://host/... hoặc rtsps://host/...")
            camera = str(cfg.get("camera", "RTSP-CAMERA"))[:80]
        else:
            raise ValueError("Nguồn hỗ trợ: upload, webcam hoặc rtsp")

        if not INFERENCE_LOCK.acquire(blocking=False):
            raise RuntimeError("Một luồng đang inference; hãy dừng luồng hiện tại trước khi mở luồng khác.")
        acquired = True
        model = await asyncio.to_thread(load_model)
        await asyncio.to_thread(reset_tracker, model)
        browser_mode = kind == "browser_webcam"
        cap = None if browser_mode else await asyncio.to_thread(_open_capture, kind, source_arg)
        if cap is not None and not cap.isOpened():
            if kind == "rtsp":
                raise RuntimeError("Không mở được RTSP qua OpenCV/FFmpeg TCP. Kiểm tra host/port/path, tài khoản và mật khẩu (URL-encode ký tự đặc biệt), quyền truy cập mạng từ máy chạy backend và codec của camera.")
            if kind == "upload":
                raise RuntimeError("OpenCV không đọc được video. Thử MP4 H.264 hoặc xác nhận codec được FFmpeg hỗ trợ.")
            raise RuntimeError("Không mở được nguồn video. Kiểm tra quyền camera, URL RTSP hoặc codec file.")
        if job_id:
            with db() as c:
                c.execute("UPDATE jobs SET status='processing',detail='Đang phân tích bằng model cục bộ.' WHERE id=?", (job_id,))
        await ws.send_json({"type": "status", "state": "running", "stream_id": stream_id, "camera": camera, "device": device_info()})
        candidate_frames: dict[int, int] = {}
        candidate_best: dict[int, tuple[float, np.ndarray, np.ndarray, list[float]]] = {}
        created_tracks: set[int] = set()
        event_count = 0
        frame_no = 0
        reconnects = 0
        started = time.perf_counter()
        while True:
            if browser_mode:
                payload = await ws.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ok = True
            else:
                try:
                    ok, frame = await asyncio.to_thread(cap.read)
                except Exception as exc:
                    raise RuntimeError(f"Không đọc được frame: {exc}") from exc
                if not ok:
                    if kind == "upload":
                        break
                    reconnects += 1
                    if reconnects > 5:
                        raise RuntimeError("Nguồn live mất kết nối sau 5 lần thử phục hồi.")
                    await ws.send_json({"type":"status", "state":"reconnecting", "attempt":reconnects})
                    await asyncio.to_thread(cap.release)
                    await asyncio.sleep(min(0.5 * (2 ** (reconnects - 1)), 8))
                    cap = await asyncio.to_thread(_open_capture, kind, source_arg)
                    if not cap.isOpened():
                        continue
                    continue
                reconnects = 0
            frame_no += 1
            result = await asyncio.to_thread(infer_track, model, frame, float(cfg.get("confidence", 0.35)))
            annotated = _draw_detections(frame, result)
            boxes = result.boxes
            detections = []
            frame_candidates = {}
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                motor_boxes = [xyxy[i] for i,cid in enumerate(clss) if cid == 0]
                for i, box in enumerate(xyxy):
                    cid, tid, score = int(clss[i]), ids[i], float(confs[i])
                    detections.append({"class": LABELS.get(cid, str(cid)), "confidence": round(score, 3), "tracking_id": tid, "box": [round(v,1) for v in box]})
                    if cid != 3 or tid is None:
                        continue
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    related = any(mb[0] <= cx <= mb[2] and mb[1] <= cy <= mb[3] for mb in motor_boxes)
                    if related:
                        tid = int(tid)
                        frame_candidates[tid] = max(frame_candidates.get(tid, 0), score)
                        if tid not in created_tracks:
                            if tid not in candidate_best or score > candidate_best[tid][0]:
                                candidate_best[tid] = (score, frame.copy(), annotated.copy(), box)
                            if candidate_frames.get(tid, 0) + 1 >= EVENT_CONFIRM_FRAMES:
                                best_score, best_frame, best_annotated, best_box = candidate_best.pop(tid)
                                event = await asyncio.to_thread(_save_event, best_frame, best_annotated, best_box, best_score, camera, stream_id, tid)
                                created_tracks.add(tid)
                                event_count += 1
                                await ws.send_json({"type":"event", **event, "evidence_url":f"/api/events/{event['id']}/evidence/annotated"})
            for tid in list(candidate_frames):
                if tid not in frame_candidates:
                    candidate_frames[tid] = 0
                    candidate_best.pop(tid, None)
            for tid in frame_candidates:
                candidate_frames[tid] = candidate_frames.get(tid, 0) + 1
            output_frame = annotated if cfg.get("overlay", True) else frame
            width = output_frame.shape[1]
            if width > 1280:
                scale = 1280 / width
                output_frame = cv2.resize(output_frame, (1280, int(output_frame.shape[0]*scale)))
            ok, encoded = await asyncio.to_thread(cv2.imencode, ".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                elapsed = max(time.perf_counter()-started, 0.001)
                await ws.send_json({"type":"frame", "frame":frame_no, "fps":round(frame_no/elapsed,1), "detections":detections, "image":base64.b64encode(encoded).decode("ascii")})
            if kind not in {"upload", "browser_webcam"}:
                await asyncio.sleep(0.001)
        if job_id:
            with db() as c:
                c.execute("UPDATE jobs SET status='completed',detail=? WHERE id=?", (f"Đã xử lý {frame_no} frame, tạo {event_count} sự kiện cần rà soát.", job_id))
            job_finished = True
        await ws.send_json({"type":"status", "state":"completed", "frames":frame_no, "events":event_count})
    except WebSocketDisconnect:
        if job_id and not job_finished:
            with db() as c:
                c.execute("UPDATE jobs SET status='cancelled',detail='Luồng đã dừng trước khi video xử lý xong.' WHERE id=?", (job_id,))
    except Exception as exc:
        try:
            await ws.send_json({"type":"error", "message":str(exc)})
        except Exception:
            pass
        if job_id:
            with db() as c:
                c.execute("UPDATE jobs SET status='failed',detail=? WHERE id=?", (str(exc)[:1000], job_id))
    finally:
        if cap is not None:
            await asyncio.to_thread(cap.release)
        if acquired:
            INFERENCE_LOCK.release()


@app.patch("/api/events/{event_id}/review")
def review(event_id: str, payload: Review):
    with db() as c:
        cur = c.execute("UPDATE events SET status=?, note=? WHERE id=?", (payload.status, payload.note, event_id))
        if not cur.rowcount:
            raise HTTPException(404, "Không tìm thấy sự kiện")
        c.execute("CREATE TABLE IF NOT EXISTS audit_logs (id TEXT PRIMARY KEY, event_id TEXT, reviewer TEXT, action TEXT, note TEXT, created_at TEXT)")
        c.execute("INSERT INTO audit_logs VALUES (?,?,?,?,?,?)", (str(uuid.uuid4()), event_id, payload.reviewer, payload.status, payload.note, datetime.now(timezone.utc).isoformat()))
    return {"ok": True, "event_id": event_id, "status": payload.status}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(415, "Định dạng video được hỗ trợ: mp4, avi, mov, mkv, webm")
    job_id = str(uuid.uuid4())
    target = UPLOADS / f"{job_id}{suffix}"
    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 500 * 1024 * 1024:
                target.unlink(missing_ok=True)
                raise HTTPException(413, "Video vượt giới hạn 500 MB")
            out.write(chunk)
    model = model_status()
    state = "queued" if model["loaded"] else "blocked"
    detail = "Video đã nhận; mở luồng WebSocket để bắt đầu inference." if model["loaded"] else model["error"]
    with db() as c:
        c.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (job_id, Path(file.filename or target.name).name, datetime.now(timezone.utc).isoformat(), state, detail))
    return {"job_id": job_id, "filename": Path(file.filename or target.name).name, "status": state, "detail": detail}


@app.get("/api/jobs")
def jobs():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")]


@app.get("/api/models")
def models():
    return [{"name": MODEL_PATH.name, "version": "unknown", "loaded": model_status()["loaded"], "metrics": None, "error": model_status()["error"]}]

