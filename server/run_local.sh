#!/bin/sh
# One command: page and API on http://127.0.0.1:8000, Mac kept awake until this stops.
# Point a tunnel at that port. Ctrl-C stops everything. Keep the Mac plugged in, lid open:
# caffeinate's system-sleep guard only holds on AC power, and closing the lid can still sleep it.
#
#   server/run_local.sh                     KSHANA_THREADS and KSHANA_PORT override.
#   KSHANA_LOW_MEMORY=1 server/run_local.sh loads each model per run: far less memory, slightly slower.
#   KSHANA_CORS_ORIGIN=https://x server/run_local.sh    only if the page is hosted elsewhere.
set -eu
cd "$(dirname "$0")/.."

PYTHON=${KSHANA_PYTHON:-/opt/miniconda3/envs/kshana/bin/python}
if [ ! -f server/models/unet.onnx.data ]; then
  echo "No weights in server/models. Run: PYTHONPATH=. $PYTHON optimize/export_openvino.py --out server/models" >&2
  exit 1
fi

export KSHANA_THREADS="${KSHANA_THREADS:-4}"
export OMP_NUM_THREADS="$KSHANA_THREADS"
export KSHANA_PORT="${KSHANA_PORT:-8000}"
export KSHANA_BEHIND_CLOUDFLARE=1
export PYTHONDONTWRITEBYTECODE=1
# Stops macOS keeping freed pages resident.
export MallocMediumZone=0 MallocLargeCache=0

# Blocks torch: spaCy would otherwise import it, for about 200 MB this server never uses.
"$PYTHON" -c '
import os, sys, uvicorn
sys.modules["torch"] = None
uvicorn.run("server.app:create_app", factory=True, host="127.0.0.1", port=int(os.environ["KSHANA_PORT"]), log_level="warning")
' &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null' INT TERM
caffeinate -dims -w "$SERVER" &
echo "Kshana is starting on http://127.0.0.1:$KSHANA_PORT with $KSHANA_THREADS threads. Ctrl-C stops it."
wait "$SERVER" || true
wait "$SERVER" 2>/dev/null || true
