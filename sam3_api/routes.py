from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from .config import MAX_VIDEO_SECONDS, SESSION_TTL_SECONDS
from .image_utils import mask_to_png, overlay_frame, rgb_to_png
from .schemas import (
    AddBBoxResponse,
    BBoxRequest,
    CreateSessionResponse,
    ObjectBoxesResponse,
    ProgressResponse,
    PropagateRequest,
    StatusResponse,
)
from .service import Sam3Service

router = APIRouter(prefix="/api")
_service: Sam3Service | None = None


def bind_service(service: Sam3Service) -> None:
    global _service
    _service = service


def _get_service() -> Sam3Service:
    if _service is None:
        raise HTTPException(status_code=500, detail="Web API service is not initialized.")
    return _service


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(video: UploadFile = File(...)) -> CreateSessionResponse:
    service = _get_service()
    data = await video.read()
    session = service.create_session_from_video(data, video.filename or "video.mp4")
    return CreateSessionResponse(
        session_id=session.backend_session_id,
        frame_count=session.frame_count,
        width=session.width,
        height=session.height,
        max_video_seconds=MAX_VIDEO_SECONDS,
        ttl_seconds=SESSION_TTL_SECONDS,
    )


@router.post("/sessions/{session_id}/bbox", response_model=AddBBoxResponse)
async def add_bbox(session_id: str, bbox: BBoxRequest) -> AddBBoxResponse:
    service = _get_service()
    session = service.get_session(session_id)
    payload = service.add_bbox_prompt(session, bbox)
    return AddBBoxResponse(**payload)


@router.get("/sessions/{session_id}/mask/{frame_index}")
async def get_mask(session_id: str, frame_index: int) -> Response:
    service = _get_service()
    session = service.get_session(session_id)
    with session.lock:
        mask = session.masks_by_frame.get(frame_index)
    if mask is None:
        raise HTTPException(status_code=404, detail=f"Mask for frame {frame_index} not found.")
    return Response(content=mask_to_png(mask), media_type="image/png")


@router.get("/sessions/{session_id}/object-boxes", response_model=ObjectBoxesResponse)
async def get_object_boxes(session_id: str) -> ObjectBoxesResponse:
    service = _get_service()
    session = service.get_session(session_id)
    return ObjectBoxesResponse(**service.build_object_boxes_export(session))


@router.get("/sessions/{session_id}/frames/{frame_index}")
async def get_frame(session_id: str, frame_index: int, overlay: bool = False) -> Response:
    service = _get_service()
    session = service.get_session(session_id)
    with session.lock:
        if frame_index < 0 or frame_index >= session.frame_count:
            raise HTTPException(status_code=400, detail="frame_index is out of range.")
        frame_rgb = session.frames_rgb[frame_index]
        if overlay:
            mask = session.masks_by_frame.get(frame_index)
            if mask is not None:
                frame_rgb = overlay_frame(frame_rgb, mask)
    return Response(content=rgb_to_png(frame_rgb), media_type="image/png")


@router.post("/sessions/{session_id}/propagate/start", response_model=StatusResponse)
async def start_propagation(
    session_id: str,
    body: PropagateRequest = PropagateRequest(),
) -> StatusResponse:
    service = _get_service()
    session = service.get_session(session_id)
    await service.start_propagation(session, selected_labels=body.selected_labels)
    return StatusResponse(status="started")


@router.post("/sessions/{session_id}/propagate/cancel", response_model=StatusResponse)
async def cancel_propagation(session_id: str) -> StatusResponse:
    service = _get_service()
    session = service.get_session(session_id)
    with session.lock:
        if session.propagation_task and not session.propagation_task.done():
            session.propagation_task.cancel()
            service.predictor.handle_request(
                {"type": "cancel_propagation", "session_id": session.predictor_session_id}
            )
            session.propagation_status = "cancelled"
            session.touch()
    return StatusResponse(status="cancelled")


@router.get("/sessions/{session_id}/progress", response_model=ProgressResponse)
async def get_progress(session_id: str) -> ProgressResponse:
    service = _get_service()
    session = service.get_session(session_id)
    with session.lock:
        total = session.frame_count
        processed = min(session.processed_frames, total)
        progress = processed / total if total else 0.0
        return ProgressResponse(
            session_id=session.backend_session_id,
            status=session.propagation_status,
            processed_frames=processed,
            total_frames=total,
            progress=progress,
            error=session.error,
        )


@router.post("/sessions/{session_id}/close", response_model=StatusResponse)
async def close_session(session_id: str) -> StatusResponse:
    service = _get_service()
    was_closed = service.close_session(session_id)
    if not was_closed:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return StatusResponse(status="closed")
