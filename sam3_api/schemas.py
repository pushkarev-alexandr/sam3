from typing import Literal, Optional

from pydantic import BaseModel, Field


class BBoxRequest(BaseModel):
    frame_index: int = Field(ge=0)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    polarity: Literal["positive", "negative"] = "positive"
    mode: Literal["replace_last", "append"] = "replace_last"


class ProgressResponse(BaseModel):
    session_id: str
    status: str
    processed_frames: int
    total_frames: int
    progress: float
    error: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    frame_count: int
    width: int
    height: int
    max_video_seconds: int
    ttl_seconds: int


class AddBBoxResponse(BaseModel):
    frame_index: int
    mask_ready: bool
    width: int
    height: int
    object_ids: list[int]
    prompt_boxes: list["PromptBox"]


class PromptBox(BaseModel):
    frame_index: int
    x: float
    y: float
    width: float
    height: float
    polarity: Literal["positive", "negative"]


class ObjectBox(BaseModel):
    label: int
    model_object_id: int | None = None
    box_xywh: list[float]


class FrameObjectBoxes(BaseModel):
    frame_index: int
    objects: list[ObjectBox]


class ObjectBoxesResponse(BaseModel):
    version: int
    coordinate_space: str
    frames: list[FrameObjectBoxes]


class PropagateRequest(BaseModel):
    selected_labels: list[int] | None = None


class StatusResponse(BaseModel):
    status: str
