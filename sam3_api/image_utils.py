from io import BytesIO
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import MASK_ALPHA, MASK_COLOR


def merge_output_masks(raw_masks: object) -> Optional[np.ndarray]:
    if raw_masks is None:
        return None
    masks = raw_masks
    if isinstance(masks, list):
        masks = np.array(masks)
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        return None
    return np.any(masks.astype(bool), axis=0)


def normalize_obj_ids(obj_ids: object) -> List[int]:
    if obj_ids is None:
        return []
    if hasattr(obj_ids, "cpu"):
        obj_ids = obj_ids.cpu().numpy()
    return [int(x) for x in np.asarray(obj_ids).reshape(-1).tolist()]


def mask_to_png(mask: np.ndarray) -> bytes:
    data = mask.astype(np.uint8) * 255
    image = Image.fromarray(data, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overlay_frame(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame_rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    out[mask_bool] = out[mask_bool] * (1.0 - MASK_ALPHA) + MASK_COLOR * MASK_ALPHA
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_to_png(frame_rgb: np.ndarray) -> bytes:
    image = Image.fromarray(frame_rgb, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
