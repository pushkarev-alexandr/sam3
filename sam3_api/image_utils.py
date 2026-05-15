from io import BytesIO
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import MASK_ALPHA


MASK_COLOR_PALETTE = np.array(
    [
        [255, 59, 48],
        [0, 122, 255],
        [52, 199, 89],
        [255, 149, 0],
        [175, 82, 222],
        [255, 45, 85],
        [90, 200, 250],
        [255, 204, 0],
        [88, 86, 214],
        [48, 176, 199],
        [162, 132, 94],
        [255, 159, 243],
    ],
    dtype=np.uint8,
)


def _normalize_output_masks(raw_masks: object) -> Optional[np.ndarray]:
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
    return masks.astype(bool)


def merge_output_masks(raw_masks: object) -> Optional[np.ndarray]:
    masks = _normalize_output_masks(raw_masks)
    if masks is None:
        return None
    return np.any(masks, axis=0)


def normalize_obj_ids(obj_ids: object) -> List[int]:
    if obj_ids is None:
        return []
    if hasattr(obj_ids, "cpu"):
        obj_ids = obj_ids.cpu().numpy()
    return [int(x) for x in np.asarray(obj_ids).reshape(-1).tolist()]


def build_label_mask(raw_masks: object, _obj_ids: object) -> Optional[np.ndarray]:
    """Build per-pixel labels 1..N from N instance masks (one distinct color each).

    Uses mask row index, not raw ``object_id`` from the model, so ids like ``[0,1,2]``
    never collapse two instances onto the same label (``0`` and ``1`` both mapped to 1).

    ``_obj_ids`` is accepted for API compatibility with callers but not used for labeling.

    Overlap: at each pixel, label is the maximum (1-based) mask index among masks
    that cover the pixel (same spirit as "higher id wins", but stable for visualization).
    """
    masks = _normalize_output_masks(raw_masks)
    if masks is None:
        return None

    n = int(masks.shape[0])
    labels = np.zeros(masks.shape[1:], dtype=np.uint16)
    for i in range(n):
        m = masks[i]
        lab = i + 1
        labels = np.where(m, np.maximum(labels, lab), labels)
    return np.minimum(labels, 255).astype(np.uint8)


def filter_label_mask(label_mask: np.ndarray, selected_labels: list[int]) -> np.ndarray:
    if not selected_labels:
        return label_mask
    keep = np.zeros_like(label_mask, dtype=bool)
    for lab in selected_labels:
        keep |= label_mask == lab
    result = label_mask.copy()
    result[~keep] = 0
    return result


def filter_object_boxes(
    objects: list[dict[str, object]],
    selected_labels: list[int],
) -> list[dict[str, object]]:
    if not selected_labels:
        return objects
    keep = set(selected_labels)
    return [obj for obj in objects if int(obj.get("label", 0)) in keep]


def color_for_object_id(object_id: int) -> np.ndarray:
    if object_id <= 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    return MASK_COLOR_PALETTE[(object_id - 1) % len(MASK_COLOR_PALETTE)]


def mask_to_png(mask: np.ndarray) -> bytes:
    if mask.dtype == bool:
        data = mask.astype(np.uint8) * 255
    else:
        data = mask.astype(np.uint8)
    image = Image.fromarray(data, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overlay_frame(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame_rgb.astype(np.float32).copy()
    label_mask = mask.astype(np.uint8)
    for object_id in np.unique(label_mask):
        if object_id == 0:
            continue
        mask_bool = label_mask == object_id
        color = color_for_object_id(int(object_id)).astype(np.float32)
        out[mask_bool] = out[mask_bool] * (1.0 - MASK_ALPHA) + color * MASK_ALPHA
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_to_png(frame_rgb: np.ndarray) -> bytes:
    image = Image.fromarray(frame_rgb, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
