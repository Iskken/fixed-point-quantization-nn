#!/usr/bin/env bash
# The layer-wise method at 4/4 on the fresh seeds 1-4 (PLAN.md, "Layer-wise controls at 4/4"), at the setting of
# the best seed-0 run (lr 3e-4, ramp 2, alpha_end 3, beta 2; REPORT.md section 10):
#   c: the exact staircase in layer k (the method);
#   d: the STE control (--current-layer ste): layer k rounds and uses slope 1, like every other trained layer.
# The two alternate seed by seed, so a stop at any point leaves complete pairs. Group "main": the test set is
# evaluated once, after the last stage. Compare c with d (paired, seeds 1-4) and with the plain STE (main group,
# lr 1e-2, 120 epochs: test 91.31 +- 0.12 after re-estimation; results/qat/main/summary_w4a4.json).
#
# Run from the repository root:
#   start:   setsid nohup bash piecewise_nonlinearity/run_layerwise_seeds_4w4a.sh > /dev/null 2>&1 < /dev/null &
#   watch:   tail -f piecewise_nonlinearity/results/raw/qat/layerwise_seeds_4w4a.log
#   stop:    pkill -f "run_layerwise_seeds_4w4a|piecewise_nonlinearity\.train_layerwise"
#   resume:  start it again: finished runs are skipped, the interrupted one continues after its last epoch
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=piecewise_nonlinearity/results/raw/qat/layerwise_seeds_4w4a.log
COMMON=(--group main --bits 4/4 --lr 3e-4 --stage-ramp-epochs 2 --alpha-end 3 --beta 2)
mkdir -p "$(dirname "$LOG")"

if pgrep -f "piecewise_nonlinearity\.(train_qat|train_float|train_layerwise|tune|fresh_seeds|explore)" > /dev/null; then
  echo "$(date '+%F %T') not started: another training job is running" | tee -a "$LOG"
  exit 1
fi

echo "$(date '+%F %T') start: layer-wise at 4/4, seeds 1-4, exact staircase and STE control" >> "$LOG"
for seed in 1 2 3 4; do
  for variant in "--current-layer staircase" "--current-layer ste"; do
    echo "$(date '+%F %T') run seed $seed $variant" >> "$LOG"
    # shellcheck disable=SC2086
    $PY -m piecewise_nonlinearity.train_layerwise "${COMMON[@]}" --seed "$seed" $variant >> "$LOG" 2>&1
    code=$?
    if [ "$code" -gt 128 ]; then echo "$(date '+%F %T') stopped by a signal" >> "$LOG"; exit 0; fi
    if [ "$code" -ne 0 ]; then echo "$(date '+%F %T') seed $seed $variant FAILED (exit $code)" >> "$LOG"; fi
  done
done
echo "$(date '+%F %T') layer-wise at 4/4 on seeds 1-4 finished" >> "$LOG"
