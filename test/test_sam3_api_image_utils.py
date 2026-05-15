from io import BytesIO
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

SAM3_API_DIR = Path(__file__).resolve().parents[1] / "sam3_api"
sam3_api_package = types.ModuleType("sam3_api")
sam3_api_package.__path__ = [str(SAM3_API_DIR)]
sys.modules.setdefault("sam3_api", sam3_api_package)

spec = importlib.util.spec_from_file_location("sam3_api.image_utils", SAM3_API_DIR / "image_utils.py")
assert spec is not None and spec.loader is not None
image_utils = importlib.util.module_from_spec(spec)
sys.modules["sam3_api.image_utils"] = image_utils
spec.loader.exec_module(image_utils)

build_label_mask = image_utils.build_label_mask
build_mask_for_model_ids = image_utils.build_mask_for_model_ids
mask_to_png = image_utils.mask_to_png
overlay_frame = image_utils.overlay_frame


def test_build_mask_for_model_ids_keeps_only_selected_track() -> None:
    masks = np.array(
        [
            [[True, False], [False, False]],
            [[False, True], [True, True]],
        ]
    )

    mask = build_mask_for_model_ids(masks, [0, 1], [0])

    assert mask.tolist() == [[1, 0], [0, 0]]


def test_build_mask_for_model_ids_empty_when_track_absent() -> None:
    masks = np.array([[[False, True], [True, True]]])

    mask = build_mask_for_model_ids(masks, [1], [0])

    assert mask.tolist() == [[0, 0], [0, 0]]


def test_build_label_mask_uses_mask_index_and_max_index_on_overlap() -> None:
    masks = np.array(
        [
            [[True, True], [False, False]],
            [[False, True], [True, False]],
        ]
    )

    label_mask = build_label_mask(masks, [2, 5])

    assert label_mask is not None
    assert label_mask.tolist() == [[1, 2], [2, 0]]


def test_three_disjoint_masks_get_labels_1_2_3_even_if_object_ids_start_at_zero() -> None:
    masks = np.array(
        [
            [[True, False, False]],
            [[False, True, False]],
            [[False, False, True]],
        ]
    )

    label_mask = build_label_mask(masks, [0, 1, 2])

    assert label_mask is not None
    assert label_mask.tolist() == [[1, 2, 3]]


def test_mask_to_png_preserves_label_values() -> None:
    label_mask = np.array([[0, 2], [5, 0]], dtype=np.uint8)

    with Image.open(BytesIO(mask_to_png(label_mask))) as image:
        decoded = np.asarray(image)

    assert decoded.tolist() == [[0, 2], [5, 0]]


def test_overlay_frame_uses_different_colors_per_label() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    label_mask = np.array([[0, 2], [5, 0]], dtype=np.uint8)

    overlay = overlay_frame(frame, label_mask)

    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[0, 1].tolist() != [0, 0, 0]
    assert overlay[1, 0].tolist() != overlay[0, 1].tolist()
