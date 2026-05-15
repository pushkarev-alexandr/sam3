from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import HTTPException

from sam3 import build_sam3_predictor

from .config import (
    MAX_UPLOAD_BYTES,
    MAX_VIDEO_FRAMES,
    MAX_VIDEO_SECONDS,
    SESSION_TTL_SECONDS,
    TEMP_DIR_NAME,
)
from .image_utils import build_label_mask, filter_label_mask, filter_object_boxes, normalize_obj_ids
from .schemas import BBoxRequest


@dataclass
class SessionState:
    backend_session_id: str
    predictor_session_id: str
    temp_dir: Path
    video_path: Path
    frames_dir: Path
    frame_count: int
    width: int
    height: int
    frames_rgb: List[np.ndarray]
    first_prompt_frame_index: Optional[int] = None
    masks_by_frame: Dict[int, np.ndarray] = field(default_factory=dict)
    object_boxes_by_frame: Dict[int, list[dict[str, object]]] = field(default_factory=dict)
    propagation_status: str = "idle"
    processed_frames: int = 0
    error: Optional[str] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    propagation_task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


class Sam3Service:
    def __init__(self) -> None:
        self.predictor = build_sam3_predictor(
            version=os.getenv("SAM3_VERSION", "sam3"),
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            apply_temporal_disambiguation=False,
        )
        self.sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def create_session_from_video(self, video_bytes: bytes, filename: str) -> SessionState:
        if len(video_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Video is too large. Max allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        temp_root = Path.cwd() / TEMP_DIR_NAME
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3-web-", dir=temp_root))
        video_path = temp_dir / (filename or "video.mp4")
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(video_bytes)

        frame_count, width, height, frames_rgb = self._extract_frames(video_path, frames_dir)
        if frame_count == 0:
            self._safe_rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="Unable to decode frames from video.")
        if frame_count > MAX_VIDEO_FRAMES:
            self._safe_rmtree(temp_dir)
            raise HTTPException(
                status_code=400,
                detail=f"Video has {frame_count} frames, limit is {MAX_VIDEO_FRAMES}.",
            )

        predictor_response = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(video_path)}
        )
        predictor_session_id = predictor_response["session_id"]

        backend_session_id = str(uuid.uuid4())
        state = SessionState(
            backend_session_id=backend_session_id,
            predictor_session_id=predictor_session_id,
            temp_dir=temp_dir,
            video_path=video_path,
            frames_dir=frames_dir,
            frame_count=frame_count,
            width=width,
            height=height,
            frames_rgb=frames_rgb,
        )
        with self._lock:
            self.sessions[backend_session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        session.touch()
        return session

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        with contextlib.suppress(Exception):
            self.predictor.handle_request(
                {"type": "close_session", "session_id": session.predictor_session_id}
            )
        if session.propagation_task is not None:
            session.propagation_task.cancel()
        self._safe_rmtree(session.temp_dir)
        return True

    async def close_expired_sessions(self) -> None:
        now = time.time()
        with self._lock:
            expired_ids = [
                sid
                for sid, session in self.sessions.items()
                if now - session.updated_at > SESSION_TTL_SECONDS
            ]
        for sid in expired_ids:
            self.close_session(sid)

    def add_bbox_prompt(self, session: SessionState, bbox: BBoxRequest) -> Dict[str, object]:
        with session.lock:
            if bbox.frame_index >= session.frame_count:
                raise HTTPException(
                    status_code=400,
                    detail=f"frame_index={bbox.frame_index} is out of range [0, {session.frame_count - 1}]",
                )
            session.first_prompt_frame_index = bbox.frame_index
            box_xywh = self._normalize_bbox_xywh(
                x=bbox.x,
                y=bbox.y,
                width=bbox.width,
                height=bbox.height,
                frame_width=session.width,
                frame_height=session.height,
            )
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session.predictor_session_id,
                    "frame_index": bbox.frame_index,
                    "bounding_boxes": [box_xywh],
                    "bounding_box_labels": [1],
                    "rel_coordinates": True,
                    "clear_old_boxes": True,
                }
            )
            outputs = response.get("outputs", {})
            label_mask = build_label_mask(
                outputs.get("out_binary_masks"),
                outputs.get("out_obj_ids"),
            )
            if label_mask is None or not np.any(label_mask):
                raise HTTPException(
                    status_code=500,
                    detail="Model did not return a mask for bbox prompt.",
                )
            session.masks_by_frame[bbox.frame_index] = label_mask
            session.object_boxes_by_frame[bbox.frame_index] = self._build_frame_object_boxes(
                outputs=outputs,
            )
            session.touch()
            return {
                "frame_index": bbox.frame_index,
                "mask_ready": True,
                "width": int(label_mask.shape[1]),
                "height": int(label_mask.shape[0]),
                "object_ids": normalize_obj_ids(outputs.get("out_obj_ids")),
            }

    async def start_propagation(
        self,
        session: SessionState,
        selected_labels: list[int] | None = None,
    ) -> None:
        with session.lock:
            if session.first_prompt_frame_index is None:
                raise HTTPException(
                    status_code=400,
                    detail="Add bbox prompt first before starting propagation.",
                )
            if session.propagation_task and not session.propagation_task.done():
                raise HTTPException(status_code=409, detail="Propagation is already running.")
            session.propagation_status = "running"
            session.processed_frames = 0
            session.error = None
            session.touch()
            session.propagation_task = asyncio.create_task(
                self._run_propagation(session, selected_labels=selected_labels),
                name=f"propagate-{session.backend_session_id}",
            )

    async def _run_propagation(
        self,
        session: SessionState,
        selected_labels: list[int] | None = None,
    ) -> None:
        try:
            def _run_propagation_stream() -> None:
                # to_thread runs in a different thread, so we must re-enter
                # inference/autocast contexts explicitly there.
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for response in self.predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session.predictor_session_id,
                                "start_frame_index": session.first_prompt_frame_index,
                                "propagation_direction": "both",
                            }
                        ):
                            frame_idx = response.get("frame_index")
                            if frame_idx is None:
                                continue
                            outputs = response.get("outputs", {})
                            mask = build_label_mask(
                                outputs.get("out_binary_masks"),
                                outputs.get("out_obj_ids"),
                            )
                            if mask is not None and selected_labels:
                                mask = filter_label_mask(mask, selected_labels)
                            object_boxes = self._build_frame_object_boxes(outputs=outputs)
                            if selected_labels:
                                object_boxes = filter_object_boxes(object_boxes, selected_labels)
                            with session.lock:
                                if mask is not None:
                                    session.masks_by_frame[int(frame_idx)] = mask
                                session.object_boxes_by_frame[int(frame_idx)] = object_boxes
                                session.processed_frames = min(
                                    len(session.masks_by_frame), session.frame_count
                                )
                                session.touch()

            await asyncio.to_thread(_run_propagation_stream)
            with session.lock:
                session.propagation_status = "completed"
                session.processed_frames = min(len(session.masks_by_frame), session.frame_count)
                session.touch()
        except asyncio.CancelledError:
            with session.lock:
                session.propagation_status = "cancelled"
                session.touch()
            raise
        except Exception as exc:
            with session.lock:
                session.propagation_status = "failed"
                session.error = str(exc)
                session.touch()

    def build_object_boxes_export(self, session: SessionState) -> dict[str, object]:
        with session.lock:
            frames = [
                {
                    "frame_index": frame_index,
                    "objects": session.object_boxes_by_frame.get(frame_index, []),
                }
                for frame_index in range(session.frame_count)
            ]
        return {
            "version": 1,
            "coordinate_space": "normalized_xywh",
            "frames": frames,
        }

    @staticmethod
    def _build_frame_object_boxes(*, outputs: dict[str, Any]) -> list[dict[str, object]]:
        boxes = outputs.get("out_boxes_xywh")
        if boxes is None:
            return []
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        boxes_array = np.asarray(boxes, dtype=np.float32)
        if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
            return []

        object_ids = normalize_obj_ids(outputs.get("out_obj_ids"))
        frame_objects: list[dict[str, object]] = []
        for idx, box in enumerate(boxes_array):
            frame_objects.append(
                {
                    "label": idx + 1,
                    "model_object_id": object_ids[idx] if idx < len(object_ids) else None,
                    "box_xywh": [float(value) for value in box.tolist()],
                }
            )
        return frame_objects

    @staticmethod
    def _extract_frames(
        video_path: Path, frames_dir: Path
    ) -> Tuple[int, int, int, List[np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open uploaded video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        est_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and est_frames > 0 and (est_frames / fps) > MAX_VIDEO_SECONDS:
            cap.release()
            raise HTTPException(
                status_code=400,
                detail=f"Video is too long. Max duration is {MAX_VIDEO_SECONDS} seconds.",
            )
        idx = 0
        width = 0
        height = 0
        frames_rgb: List[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            frames_rgb.append(frame_rgb)
            cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame_bgr)
            idx += 1
        cap.release()
        return idx, width, height, frames_rgb

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        with contextlib.suppress(Exception):
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _normalize_bbox_xywh(
        x: float,
        y: float,
        width: float,
        height: float,
        frame_width: int,
        frame_height: int,
    ) -> List[float]:
        if frame_width <= 0 or frame_height <= 0:
            raise HTTPException(status_code=500, detail="Invalid frame size for bbox conversion.")
        x0 = x / frame_width
        y0 = y / frame_height
        w = width / frame_width
        h = height / frame_height
        # Keep values in valid model range [0, 1].
        x0 = float(max(0.0, min(1.0, x0)))
        y0 = float(max(0.0, min(1.0, y0)))
        w = float(max(0.0, min(1.0 - x0, w)))
        h = float(max(0.0, min(1.0 - y0, h)))
        if w <= 0.0 or h <= 0.0:
            raise HTTPException(status_code=400, detail="BBox is outside frame bounds.")
        return [x0, y0, w, h]
