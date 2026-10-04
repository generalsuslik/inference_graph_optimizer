import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ONNX_DIR = Path(__file__).parent / "models" / "onnx"

DISABLE_ALL = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
EXTENDED = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ENABLE_ALL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# (label, which file, ORT level). The first two isolate what onnx-opt does: ORT's own
# rewrites are off, otherwise it folds BN and fuses Conv+Relu itself and the delta is zero.
# The last one is the reference for what ORT manages on its own.
CONFIGS = [
    ("original, ORT off", "original", DISABLE_ALL),
    ("onnx-opt, ORT off", "optimized", DISABLE_ALL),
    ("original, ORT extended", "original", EXTENDED),
    ("original, ORT all", "original", ENABLE_ALL),
]


def session(path, level, threads):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run(sess, feed):
    return sess.run(None, feed)[0]


def example_feed(name, sess):
    """The inputs export.py saved with the model, or random floats for models exported before it did."""
    saved = ONNX_DIR / f"{name}.inputs.npz"
    if saved.exists():
        with np.load(saved) as arrays:
            return {key: arrays[key] for key in arrays.files}
    rng = np.random.default_rng(0)
    return {i.name: rng.standard_normal(i.shape).astype(np.float32) for i in sess.get_inputs()}


def time_ms(sessions, feed, warmup, runs):
    """Median and p90 per session, timed round-robin so that drift in clock speed or background
    load hits every config alike instead of whichever happened to run during it."""
    for sess in sessions:
        for _ in range(warmup):
            run(sess, feed)
    samples = [[] for _ in sessions]
    for _ in range(runs):
        for sess, out in zip(sessions, samples):
            start = time.perf_counter_ns()
            run(sess, feed)
            out.append((time.perf_counter_ns() - start) / 1e6)
    return [(np.median(s), np.percentile(s, 90)) for s in samples]


def node_count(path):
    return len(onnx.load(str(path), load_external_data=False).graph.node)


def bench(name, args):
    paths = {
        "original": ONNX_DIR / f"{name}.onnx",
        "optimized": ONNX_DIR / f"{name}.opt.onnx",
    }
    # The result has to be valid ONNX, not just something onnxruntime happens to load.
    onnx.checker.check_model(str(paths["optimized"]), full_check=True)

    sessions = [session(paths[which], level, args.threads) for _, which, level in CONFIGS]
    feed = example_feed(name, sessions[0])

    # Timing a model that computes something else would be meaningless.
    reference = run(sessions[0], feed)
    optimized = run(sessions[1], feed)
    diff = float(np.abs(reference - optimized).max())
    if not np.allclose(reference, optimized, atol=1e-4, rtol=1e-4):
        raise SystemExit(f"{name}: optimized output differs from the original (max abs diff {diff:.2e})")

    print(f"\n{name}  (threads={args.threads}, {args.runs} runs, max abs diff {diff:.1e})")
    print(f"  {'config':<20} {'nodes':>6} {'median ms':>10} {'p90 ms':>8} {'speedup':>8}")
    timings = time_ms(sessions, feed, args.warmup, args.runs)
    baseline = timings[0][0]
    for (label, which, _), (median, p90) in zip(CONFIGS, timings):
        print(f"  {label:<20} {node_count(paths[which]):>6} {median:>10.2f} {p90:>8.2f} {baseline / median:>7.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Times original vs onnx-opt models under onnxruntime.")
    parser.add_argument("models", nargs="*", help="model names in models/onnx/ (default: all)")
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads, 0 = ORT default")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    names = args.models or sorted(
        p.name.removesuffix(".onnx") for p in ONNX_DIR.glob("*.onnx") if not p.name.endswith(".opt.onnx")
    )
    for name in names:
        bench(name, args)


if __name__ == "__main__":
    main()
