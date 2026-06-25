#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/review_ablations}"
COMMON=(
  "$ROOT/tests/benchmark_serving_trace.py"
  --hole-fraction 0.0
  --precompute-metadata
  --adaptive-engine
)

mkdir -p "$RESULTS_DIR"

run() {
  local name="$1"
  shift
  echo "=== $name ==="
  "$PYTHON_BIN" "${COMMON[@]}" "$@" --output "$RESULTS_DIR/$name.txt"
}

# Five held-out seeds for the headline routes.
for seed in 20260623 20260624 20260625 20260626 20260627; do
  run "seed_${seed}_bucketed_b1" \
    --trace bucketed --requests 16 --max-active 1 --steps 64 --repeats 3 \
    --seed "$seed" --adaptive-bucket-splits 32
  run "seed_${seed}_bimodal_b8" \
    --trace bimodal --requests 24 --max-active 8 --steps 48 --repeats 1 \
    --seed "$seed" --adaptive-workqueue-splits 20
  run "seed_${seed}_uniform_b8" \
    --trace uniform --requests 24 --max-active 8 --steps 48 --repeats 1 \
    --seed "$seed" --adaptive-workqueue-splits 24
  run "seed_${seed}_zipf_b8" \
    --trace zipf --requests 24 --max-active 8 --steps 48 --repeats 1 \
    --seed "$seed" --adaptive-workqueue-splits 28
done

# Split sensitivity around the calibrated operating points on the paper seed.
for splits in 28 32 36; do
  run "split_sensitivity_bucketed_b1_s${splits}" \
    --trace bucketed --requests 16 --max-active 1 --steps 64 --repeats 3 \
    --seed 20260623 --adaptive-bucket-splits "$splits"
done

for splits in 16 20 24; do
  run "split_sensitivity_bimodal_b8_s${splits}" \
    --trace bimodal --requests 24 --max-active 8 --steps 48 --repeats 1 \
    --seed 20260623 --adaptive-workqueue-splits "$splits"
done

# GQA/MQA serving sweep. Hq is fixed at 32, so hkv=32/8/4 gives G=1/4/8.
for hkv in 32 8 4; do
  g=$((32 / hkv))
  run "gqa_serving_bimodal_b8_g${g}" \
    --trace bimodal --requests 24 --max-active 8 --steps 48 --repeats 1 \
    --seed 20260623 --adaptive-workqueue-splits 20 --hkv "$hkv"
done

echo "Review ablation logs written to $RESULTS_DIR"
