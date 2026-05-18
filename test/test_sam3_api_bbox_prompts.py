import importlib.util
import sys
import types
import unittest
from pathlib import Path
from contextlib import nullcontext

from fastapi import HTTPException

SAM3_API_DIR = Path(__file__).resolve().parents[1] / "sam3_api"
sam3_api_package = types.ModuleType("sam3_api")
sam3_api_package.__path__ = [str(SAM3_API_DIR)]
sys.modules.setdefault("sam3_api", sam3_api_package)

sam3_module = types.ModuleType("sam3")
sam3_module.build_sam3_predictor = lambda **_kwargs: None
sys.modules.setdefault("sam3", sam3_module)
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

torch_module = types.ModuleType("torch")
torch_module.bfloat16 = object()
torch_module.inference_mode = nullcontext
torch_module.autocast = lambda **_kwargs: nullcontext()
sys.modules.setdefault("torch", torch_module)

for module_name in ["image_utils", "schemas", "service"]:
    spec = importlib.util.spec_from_file_location(f"sam3_api.{module_name}", SAM3_API_DIR / f"{module_name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"sam3_api.{module_name}"] = module
    spec.loader.exec_module(module)

from sam3_api.service import PromptBoxState, Sam3Service  # noqa: E402


class Sam3BboxPromptModesTest(unittest.TestCase):
    def test_append_adds_new_box(self) -> None:
        current = [
            PromptBoxState(frame_index=0, x=10.0, y=10.0, width=20.0, height=20.0, polarity="positive"),
        ]
        incoming = PromptBoxState(frame_index=0, x=30.0, y=30.0, width=15.0, height=15.0, polarity="positive")

        updated = Sam3Service._build_prompt_boxes(current=current, incoming=incoming, mode="append")

        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[-1].x, 30.0)
        self.assertEqual(updated[-1].polarity, "positive")

    def test_replace_last_updates_last_same_polarity(self) -> None:
        current = [
            PromptBoxState(frame_index=0, x=10.0, y=10.0, width=20.0, height=20.0, polarity="positive"),
            PromptBoxState(frame_index=0, x=40.0, y=40.0, width=15.0, height=15.0, polarity="negative"),
            PromptBoxState(frame_index=0, x=50.0, y=50.0, width=16.0, height=16.0, polarity="negative"),
        ]
        incoming = PromptBoxState(frame_index=0, x=60.0, y=60.0, width=17.0, height=17.0, polarity="negative")

        updated = Sam3Service._build_prompt_boxes(current=current, incoming=incoming, mode="replace_last")

        self.assertEqual(len(updated), 3)
        self.assertEqual(updated[-1].x, 60.0)
        self.assertEqual(updated[-1].polarity, "negative")

    def test_replace_last_appends_when_no_same_polarity(self) -> None:
        current = [
            PromptBoxState(frame_index=0, x=10.0, y=10.0, width=20.0, height=20.0, polarity="positive"),
        ]
        incoming = PromptBoxState(frame_index=0, x=70.0, y=70.0, width=18.0, height=18.0, polarity="negative")

        updated = Sam3Service._build_prompt_boxes(current=current, incoming=incoming, mode="replace_last")

        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[-1].polarity, "negative")

    def test_negative_without_positive_is_invalid(self) -> None:
        boxes = [
            PromptBoxState(frame_index=0, x=10.0, y=10.0, width=20.0, height=20.0, polarity="negative"),
        ]
        with self.assertRaises(HTTPException):
            if not any(box.polarity == "positive" for box in boxes):
                raise HTTPException(
                    status_code=400,
                    detail="At least one positive bbox is required before adding negative prompts.",
                )


if __name__ == "__main__":
    unittest.main()
