#!/usr/bin/env bash
# First check of the layer-by-layer staircase (train_layerwise.py) at 2/2 on seed 0, the development seed:
# step 1 of the tuning procedure (PLAN.md, change 4: the learning rate 3e-4, 1e-3, 3e-3) at the exact
# staircase's chosen setting with the learning rate tied to alpha (alpha_end 3, beta 3, lr / alpha^2;
# results/qat/tune/selection_piecewise-lrpow2_w2a2.json), with the default stages (alpha ramp of 2 epochs,
# patience 3, at most 15 epochs per stage, early stopping on validation accuracy after BatchNorm
# re-estimation, layers frozen with re-estimated statistics). Group "explore": seed 0 never evaluates the test set.
# Compare with the seed-0 validation accuracy after re-estimation: STE 90.88, exact staircase lr / alpha^2
# 84.58 (single runs differ by up to about 0.9).
#
# Run from the repository root:
#   start:   setsid nohup bash piecewise_nonlinearity/run_layerwise_check.sh > /dev/null 2>&1 < /dev/null &
#   watch:   tail -f piecewise_nonlinearity/results/raw/qat/layerwise_check.log
#   stop:    pkill -f "run_layerwise_check|piecewise_nonlinearity\.train_layerwise"
#   resume:  start it again: finished runs are skipped, the interrupted one continues after its last epoch
#            (a checkpoint is saved after every epoch)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=piecewise_nonlinearity/results/raw/qat/layerwise_check.log
mkdir -p "$(dirname "$LOG")"

if pgrep -f "piecewise_nonlinearity\.(train_qat|train_float|train_layerwise|tune|fresh_seeds|explore)" > /dev/null; then
  echo "$(date '+%F %T') not started: another training job is running" | tee -a "$LOG"
  exit 1
fi

echo "$(date '+%F %T') start: layer-wise check at 2/2, seed 0" >> "$LOG"
for lr in 3e-4 1e-3 3e-3; do
  echo "$(date '+%F %T') run lr $lr" >> "$LOG"
  $PY -m piecewise_nonlinearity.train_layerwise --group explore --bits 2/2 --seed 0 --lr "$lr" \
      --alpha-end 3 --beta 3 --lr-alpha-power 2 >> "$LOG" 2>&1
  code=$?
  if [ "$code" -gt 128 ]; then echo "$(date '+%F %T') stopped by a signal" >> "$LOG"; exit 0; fi
  if [ "$code" -ne 0 ]; then echo "$(date '+%F %T') lr $lr FAILED (exit $code)" >> "$LOG"; fi
done
echo "$(date '+%F %T') layer-wise check finished" >> "$LOG"
