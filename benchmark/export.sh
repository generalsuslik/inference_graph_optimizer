#!/usr/bin/env bash
# Exports the benchmark models to benchmark/models/onnx/.
#
# Usage: benchmark/export.sh [MODEL ...]   (MODEL is a file in benchmark/models/; no arguments: all of them)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$here/../.venv/bin/python3"

if [[ -x "$venv" ]]; then
    python="$venv"
else
    python="python3"
fi

exec "$python" "$here/export.py" "$@"
