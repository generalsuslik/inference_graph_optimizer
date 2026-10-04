import importlib
import pkgutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch


MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR = MODELS_DIR / "onnx"

# Every module in models/ is a model: its `build()` returns the torch module to export. It may
# also define `example_inputs()` returning the input tensors, or just an INPUT_SHAPE for a single
# float input, and INPUT_NAMES for the exported graph inputs.
MODELS = sorted(m.name for m in pkgutil.iter_modules([str(MODELS_DIR)]))
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)


def example_inputs(module):
    if hasattr(module, "example_inputs"):
        return tuple(module.example_inputs())
    return (torch.randn(*getattr(module, "INPUT_SHAPE", DEFAULT_INPUT_SHAPE)),)


def export(name):
    torch.manual_seed(0)
    module = importlib.import_module(f"models.{name}")
    m = module.build().eval()
    inputs = example_inputs(module)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    torch.onnx.export(
        m,
        inputs,
        path,
        input_names=getattr(module, "INPUT_NAMES", None),
        opset_version=13,
        do_constant_folding=False,
        dynamo=False,
    )

    # Keep the inputs next to the model, keyed by graph input name, so bench.py feeds it the
    # same realistic values (token ids, attention masks) rather than random floats.
    graph_inputs = [i.name for i in onnx.load(str(path), load_external_data=False).graph.input]
    np.savez(OUT_DIR / f"{name}.inputs.npz", **{k: t.numpy() for k, t in zip(graph_inputs, inputs)})
    print(f"wrote {path}")


if __name__ == "__main__":
    for name in sys.argv[1:] or MODELS:
        if name not in MODELS:
            sys.exit(f"unknown model {name!r}, expected one of: {', '.join(MODELS)}")
        export(name)
