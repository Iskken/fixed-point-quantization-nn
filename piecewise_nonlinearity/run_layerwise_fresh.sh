#!/usr/bin/env bash
# The layer-by-layer staircase (train_layerwise.py) at 2/2 on the fresh seeds 1-4, at the best setting of the
# seed-0 check (run_layerwise_check.sh, 2026-11-23): lr 1e-3, alpha_end 3, beta 3, no coupling to alpha, the
# default stages (alpha ramp of 2 epochs, patience 3, at most 15 epochs per stage, early stopping on validation
# accuracy after BatchNorm re-estimation, layers frozen with re-estimated statistics, cosine over 15 epochs
# restarted in every stage). Only step 1 of the tuning (the learning rate) was done; alpha_end and beta are the
# exact staircase's. Group "main": seeds >= 1 evaluate the test set after the last stage. Each seed starts from
# its own float model (pretrained/resnet20_cifar10_seed<N>.pt). Compare, paired by seed, with the fresh-seed STE,
# slope-1 and exact staircase runs in results/qat/main/.
#
# Run from the repository root:
#   start:   setsid nohup bash piecewise_nonlinearity/run_layerwise_fresh.sh > /dev/null 2>&1 < /dev/null &
#   watch:   tail -f piecewise_nonlinearity/results/raw/qat/layerwise_fresh.log
#   stop:    pkill -f "run_layerwise_fresh|piecewise_nonlinearity\.train_layerwise"
#   resume:  start it again: finished seeds are skipped, the interrupted one continues after its last epoch
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=piecewise_nonlinearity/results/raw/qat/layerwise_fresh.log
mkdir -p "$(dirname "$LOG")"

if pgrep -f "piecewise_nonlinearity\.(train_qat|train_float|train_layerwise|tune|fresh_seeds|explore)" > /dev/null; then
  echo "$(date '+%F %T') not started: another training job is running" | tee -a "$LOG"
  exit 1
fi

echo "$(date '+%F %T') start: layer-wise at 2/2 on the fresh seeds" >> "$LOG"
for seed in 1 2 3 4; do
  echo "$(date '+%F %T') run seed $seed" >> "$LOG"
  $PY -m piecewise_nonlinearity.train_layerwise --group main --bits 2/2 --seed "$seed" --lr 1e-3 \
      --alpha-end 3 --beta 3 >> "$LOG" 2>&1
  code=$?
  if [ "$code" -gt 128 ]; then echo "$(date '+%F %T') stopped by a signal" >> "$LOG"; exit 0; fi
  if [ "$code" -ne 0 ]; then echo "$(date '+%F %T') seed $seed FAILED (exit $code)" >> "$LOG"; fi
done
echo "$(date '+%F %T') layer-wise fresh seeds finished" >> "$LOG"
