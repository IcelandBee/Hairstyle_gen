import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = [
    "qwen_edit_changeperson.py",
    "firered_edit_changeperson.py",
    "flux2_klein_edit_changeperson.py",
]


class _DummyTorch(types.SimpleNamespace):
    def set_float32_matmul_precision(self, value):
        self.float32_matmul_precision = value


def _install_import_stubs():
    torch = _DummyTorch()
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False),
        ),
        cudnn=types.SimpleNamespace(benchmark=False),
    )
    sys.modules["torch"] = torch
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda stream: {})

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.Resampling = types.SimpleNamespace(LANCZOS=1)
    pil_module.Image = image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

    diffusers = types.ModuleType("diffusers")
    diffusers.QwenImageEditPlusPipeline = object
    diffusers.DiffusionPipeline = object
    diffusers.Flux2KleinPipeline = object
    sys.modules["diffusers"] = diffusers


def _load_script(script_name):
    _install_import_stubs()
    module_name = script_name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, ROOT / script_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaxSampleTest(unittest.TestCase):
    def test_none_keeps_all_images(self):
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                self.assertEqual(
                    module.apply_max_sample(["a.jpg", "b.jpg", "c.jpg"], None),
                    ["a.jpg", "b.jpg", "c.jpg"],
                )

    def test_positive_value_keeps_first_n_images(self):
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                self.assertEqual(
                    module.apply_max_sample(["a.jpg", "b.jpg", "c.jpg"], 2),
                    ["a.jpg", "b.jpg"],
                )

    def test_zero_keeps_no_images(self):
        for script_name in SCRIPT_NAMES:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                self.assertEqual(
                    module.apply_max_sample(["a.jpg", "b.jpg", "c.jpg"], 0),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
