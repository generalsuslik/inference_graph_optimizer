import importlib
import pkgutil
import sys
from pathlib import Path

import torch


MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR = MODELS_DIR / "onnx"

# Every module in models/ is a model: its `build()` returns the torch module to export.
MODELS = sorted(m.name for m in pkgutil.iter_modules([str(MODELS_DIR)]))


def export(name):
    m = importlib.import_module(f"models.{name}").build().eval()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    torch.onnx.export(
        m,
        torch.randn(1, 3, 224, 224),
        path,
        opset_version=13,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name in sys.argv[1:] or MODELS:
        if name not in MODELS:
            sys.exit(f"unknown model {name!r}, expected one of: {', '.join(MODELS)}")
        export(name)
