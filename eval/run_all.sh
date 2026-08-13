#!/usr/bin/env bash
# Reproduce every row of the evaluation tables in the README.
#
#   eval/run_all.sh --lfg checkpoints/lfg_seg_motion_m3n3.pt --kitti360 /data/KITTI-360
#
# Waymo and the Pi3 baseline are optional; each is evaluated only if you point at it:
#
#   --waymo /data/waymo_v2/validation      the Waymo split directory
#   --pi3   /path/to/pi3.safetensors       Pi3 weights, with the pi3 package importable
#
# Results land in eval/results/ as JSON, one file per model and dataset.
set -euo pipefail

LFG=""; KITTI360=""; WAYMO=""; PI3=""; DEVICE="cuda"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lfg)      LFG="$2";      shift 2 ;;
    --kitti360) KITTI360="$2"; shift 2 ;;
    --waymo)    WAYMO="$2";    shift 2 ;;
    --pi3)      PI3="$2";      shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$LFG" || -z "$KITTI360" ]]; then
  sed -n '2,12p' "$0" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p eval/results

missing () { echo "missing $1: $2" >&2; MISSING=1; }
MISSING=0
[[ -f "$LFG" ]]                  || missing "LFG checkpoint" "$LFG"
[[ -d "$KITTI360" ]]             || missing "KITTI-360 directory" "$KITTI360"
[[ -d "$KITTI360/data_2d_raw" ]] || missing "KITTI-360 images" "$KITTI360/data_2d_raw"
[[ -z "$WAYMO" || -d "$WAYMO/camera_image" ]] || missing "Waymo parquet" "$WAYMO/camera_image"
[[ -z "$PI3"   || -f "$PI3" ]]   || missing "Pi3 weights" "$PI3"
if [[ "$MISSING" -ne 0 ]]; then
  echo "see the Evaluation section of the README for the expected layout" >&2
  exit 1
fi

run () {   # run <model> <dataset> <data-root> <clip-list> <output-name> [checkpoint]
  local model=$1 dataset=$2 root=$3 clips=$4 name=$5 checkpoint=${6:-}
  local log
  log=$(mktemp)
  printf '%-12s %-9s ' "$model" "$dataset"
  if python evaluate.py --model "$model" --checkpoint "$checkpoint" \
      --dataset "$dataset" --data-root "$root" --clip-list "eval/clips/$clips" \
      --frame-stride 5 --device "$DEVICE" --output "eval/results/$name.json" >"$log" 2>&1; then
    echo "ok"
  else
    echo "FAILED"
    # surface why: the harness reports missing data and bad arguments on the last line
    grep -vE "^\s*$|Warning|warn" "$log" | tail -3 | sed 's/^/               /' >&2
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "$log"
}

FAILURES=0

echo "KITTI-360 — depth, semantics and trajectory"
run lfg        kitti360 "$KITTI360" kitti360_200.txt lfg        "$LFG"
run vggt       kitti360 "$KITTI360" kitti360_200.txt vggt
run da3        kitti360 "$KITTI360" kitti360_200.txt da3
run segformer  kitti360 "$KITTI360" kitti360_200.txt segformer
run maskformer kitti360 "$KITTI360" kitti360_200.txt maskformer
run static     kitti360 "$KITTI360" kitti360_200.txt static
if [[ -n "$PI3" ]]; then
  run pi3      kitti360 "$KITTI360" kitti360_200.txt pi3        "$PI3"
else
  echo "pi3          kitti360  skipped (pass --pi3 with its weights)"
fi

if [[ -n "$WAYMO" ]]; then
  echo
  echo "Waymo — depth and trajectory"
  run lfg  waymo "$WAYMO" waymo_200.txt waymo_lfg  "$LFG"
  run vggt waymo "$WAYMO" waymo_200.txt waymo_vggt
  run da3  waymo "$WAYMO" waymo_200.txt waymo_da3
  if [[ -n "$PI3" ]]; then
    run pi3 waymo "$WAYMO" waymo_200.txt waymo_pi3 "$PI3"
  else
    echo "pi3          waymo     skipped (pass --pi3 with its weights)"
  fi
fi

echo
if [[ "$FAILURES" -gt 0 ]]; then
  echo "$FAILURES run(s) failed; results for the rest are in eval/results/" >&2
  exit 1
fi
echo "results written to eval/results/"
