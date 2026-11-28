#!/usr/bin/env bash
# Two layer-wise variants at 4/4 on seed 0, at the setting of the best layer-wise staircase run (lr 3e-4, ramp 2,
# alpha_end 3, beta 2; REPORT.md section 10), each changing only layer k's discretizer (PLAN.md, "Layer-wise
# controls at 4/4"):
#   a: the STE control  (--current-layer ste): layer k rounds and uses slope 1, like every other trained layer;
#   b: slope-1 staircase (--backward ste): layer k keeps S in the forward pass, slope 1 in the backward pass.
# Compare with the exact staircase's 91.56 (validation after re-estimation). Group "explore": no test set.
#
# Run from the repository root:
#   start:   setsid nohup bash piecewise_nonlinearity/run_layerwise_controls_4w4a.sh > /dev/null 2>&1 < /dev/null &
#   watch:   tail -f piecewise_nonlinearity/results/raw/qat/layerwise_controls_4w4a.log
#   stop:    pkill -f "run_layerwise_controls_4w4a|piecewise_nonlinearity\.train_layerwise"
#   resume:  start it again: finished runs are skipped, the interrupted one continues after its last epoch
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=piecewise_nonlinearity/results/raw/qat/layerwise_controls_4w4a.log
COMMON=(--group explore --bits 4/4 --seed 0 --lr 3e-4 --stage-ramp-epochs 2 --alpha-end 3 --beta 2)
mkdir -p "$(dirname "$LOG")"

if pgrep -f "piecewise_nonlinearity\.(train_qat|train_float|train_layerwise|tune|fresh_seeds|explore)" > /dev/null; then
  echo "$(date '+%F %T') not started: another training job is running" | tee -a "$LOG"
  exit 1
fi

echo "$(date '+%F %T') start: layer-wise controls at 4/4, seed 0" >> "$LOG"
for variant in "--current-layer ste" "--backward ste"; do
  echo "$(date '+%F %T') run $variant" >> "$LOG"
  # shellcheck disable=SC2086
  $PY -m piecewise_nonlinearity.train_layerwise "${COMMON[@]}" $variant >> "$LOG" 2>&1
  code=$?
  if [ "$code" -gt 128 ]; then echo "$(date '+%F %T') stopped by a signal" >> "$LOG"; exit 0; fi
  if [ "$code" -ne 0 ]; then echo "$(date '+%F %T') $variant FAILED (exit $code)" >> "$LOG"; fi
done
echo "$(date '+%F %T') layer-wise controls finished" >> "$LOG"
