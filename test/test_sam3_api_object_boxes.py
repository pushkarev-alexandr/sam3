import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

SAM3_API_DIR = Path(__file__).resolve().parents[1] / "sam3_api"
sam3_api_package = types.ModuleType("sam3_api")
sam3_api_package.__path__ = [str(SAM3_API_DIR)]
sys.modules.setdefault("sam3_api", sam3_api_package)

sam3_module = types.ModuleType("sam3")
sam3_module.build_sam3_predictor = lambda **_kwargs: None
sys.modules.setdefault("sam3", sam3_module)

for module_name in ["image_utils", "schemas", "service"]:
    spec = importlib.util.spec_from_file_location(f"sam3_api.{module_name}", SAM3_API_DIR / f"{module_name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"sam3_api.{module_name}"] = module
    spec.loader.exec_module(module)

from sam3_api.image_utils import filter_object_boxes  # noqa: E402
from sam3_api.service import Sam3Service  # noqa: E402


class Sam3ObjectBoxesTest(unittest.TestCase):
    def test_build_frame_object_boxes_maps_labels_to_model_ids(self) -> None:
        boxes = Sam3Service._build_frame_object_boxes(
            outputs={
                "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
                "out_obj_ids": np.array([10, 11]),
            }
        )

        self.assertEqual([item["label"] for item in boxes], [1, 2])
        self.assertEqual([item["model_object_id"] for item in boxes], [10, 11])
        np.testing.assert_allclose(boxes[0]["box_xywh"], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(boxes[1]["box_xywh"], [0.5, 0.6, 0.1, 0.2])

    def test_filter_object_boxes_keeps_selected_labels_only(self) -> None:
        objects = [
            {"label": 1, "box_xywh": [0.1, 0.2, 0.3, 0.4]},
            {"label": 2, "box_xywh": [0.5, 0.6, 0.1, 0.2]},
            {"label": 3, "box_xywh": [0.0, 0.0, 0.1, 0.1]},
        ]
        filtered = filter_object_boxes(objects, [1, 3])
        self.assertEqual([item["label"] for item in filtered], [1, 3])


if __name__ == "__main__":
    unittest.main()
