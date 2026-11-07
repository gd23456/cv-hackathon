#!/bin/sh
# Finishes the remaining measurements one at a time at low priority. Safe to stop with Ctrl-C and rerun.
# Close other apps first. Takes about 30-60 minutes on an idle 16 GB Mac.
cd "$(dirname "$0")/.." || exit 1
PY=/opt/miniconda3/envs/kshana/bin/python
export PYTHONPATH=.
step() { echo "=== $(date '+%H:%M') $*"; nice -n 15 "$PY" "$@" 2>&1 | grep -v -i -E 'scikit|warn|it/s|Loading' | tail -6 | cut -c1-170; }
step eval/rewrite_test.py
step eval/candidates.py --config S1-A --split complex_test --k 8 --final-test
step eval/run.py --config B0 --split coco_100
step eval/make_compare_gallery.py
step eval/ablate.py
step eval/charts.py
step eval/export_web_report.py
step eval/write_eval_md.py
echo "=== $(date '+%H:%M') finished"
