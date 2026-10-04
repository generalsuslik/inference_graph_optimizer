#!/usr/bin/env bash
# Optimizes the benchmark models with onnx-opt and times them against the originals.
#
# Usage: benchmark/bench.sh [MODEL ...] [-- BENCH_ARGS]   (no models: all of them)
#   e.g. benchmark/bench.sh resnet50 -- --threads 4 --runs 500
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/.."
venv="$root/.venv/bin/python3"
onnx_dir="$here/models/onnx"

if [[ -x "$venv" ]]; then
    python="$venv"
else
    python="python3"
fi

models=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    models+=("$1")
    shift
done
[[ "${1:-}" == "--" ]] && shift

if [[ ${#models[@]} -eq 0 ]]; then
    for f in "$here"/models/*.py; do
        models+=("$(basename "$f" .py)")
    done
fi

cargo build --release --quiet --manifest-path "$root/Cargo.toml" -p onnx-opt

for model in "${models[@]}"; do
    [[ -f "$onnx_dir/$model.onnx" ]] || "$here/export.sh" "$model"
    "$root/target/release/onnx-opt" "$onnx_dir/$model.onnx" -o "$onnx_dir/$model.opt.onnx"
done

exec "$python" "$here/bench.py" "${models[@]}" "$@"
