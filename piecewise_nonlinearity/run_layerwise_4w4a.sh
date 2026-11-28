#!/usr/bin/env bash
# The layer-by-layer staircase (train_layerwise.py) at 4/4 on seed 0, the development seed (PLAN.md, "Layer-wise
# at 4/4"). Every run: the 4/4 exact staircase's chosen shape (alpha_end 3, beta 2, exact backward;
# results/qat/tune/selection_piecewise_w4a4.json), the learning rate not tied to alpha (--lr-alpha-power 0), every
# rate decaying by a cosine over 15 epochs from epoch 1 of every stage, patience 3, at most 15 epochs per stage,
# early stopping on validation accuracy after BatchNorm re-estimation. Group "explore": no test set.
#   phase 1: alpha ramp of 2 epochs, lr 1e-4, 3e-4, 1e-3, 3e-3;
#   phase 2: alpha ramp of 5 epochs at the phase-1 lr with the highest validation accuracy after re-estimation
#            (failed runs excluded; a tie goes to the smaller lr).
# Compare with the seed-0 validation accuracy after re-estimation: STE 91.84, slope-1 staircase 92.38, exact
# staircase 90.94, the starting network 89.42.
#
# Run from the repository root:
#   start:   setsid nohup bash piecewise_nonlinearity/run_layerwise_4w4a.sh > /dev/null 2>&1 < /dev/null &
#   watch:   tail -f piecewise_nonlinearity/results/raw/qat/layerwise_4w4a.log
#   stop:    pkill -f "run_layerwise_4w4a|piecewise_nonlinearity\.train_layerwise"
#   resume:  start it again: finished runs are skipped, the interrupted one continues after its last epoch
#            (a checkpoint is saved after every epoch)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=piecewise_nonlinearity/results/raw/qat/layerwise_4w4a.log
COMMON=(--group explore --bits 4/4 --seed 0 --alpha-end 3 --beta 2)
LRS=(1e-4 3e-4 1e-3 3e-3)
mkdir -p "$(dirname "$LOG")"

if pgrep -f "piecewise_nonlinearity\.(train_qat|train_float|train_layerwise|tune|fresh_seeds|explore)" > /dev/null; then
  echo "$(date '+%F %T') not started: another training job is running" | tee -a "$LOG"
  exit 1
fi

run() {  # one run; returns 1 if it was stopped by a signal
  echo "$(date '+%F %T') run $*" >> "$LOG"
  $PY -m piecewise_nonlinearity.train_layerwise "${COMMON[@]}" "$@" >> "$LOG" 2>&1
  local code=$?
  if [ "$code" -gt 128 ]; then echo "$(date '+%F %T') stopped by a signal" >> "$LOG"; return 1; fi
  if [ "$code" -ne 0 ]; then echo "$(date '+%F %T') $* FAILED (exit $code)" >> "$LOG"; fi
  return 0
}

echo "$(date '+%F %T') start: layer-wise at 4/4, seed 0" >> "$LOG"
echo "$(date '+%F %T') phase 1: ramp 2, lr ${LRS[*]}" >> "$LOG"
for lr in "${LRS[@]}"; do
  run --lr "$lr" --stage-ramp-epochs 2 || exit 0
done

best=$($PY - "${COMMON[@]}" -- "${LRS[@]}" 2>> "$LOG" <<'EOF'
import json, os, sys
from piecewise_nonlinearity import train_layerwise as lw
split = sys.argv.index("--")
common, lrs = sys.argv[1:split], sys.argv[split + 1:]
found = []
for lr in lrs:
    a = lw.parse_args([*common, "--lr", lr, "--stage-ramp-epochs", "2"])
    path = os.path.join(a.results_dir, a.group, lw.run_name(a) + ".json")
    if os.path.exists(path):
        r = json.load(open(path))
        if not r["failed"]:
            found.append((r["final"]["val_acc_reest"], -float(lr), lr))
            print(f"  lr {lr}: {r['final']['val_acc_reest']:.2f}", file=sys.stderr)
print(max(found)[2] if found else "")
EOF
)
if [ -z "$best" ]; then
  echo "$(date '+%F %T') phase 2 not started: no phase-1 run finished without failing" >> "$LOG"
  exit 1
fi
echo "$(date '+%F %T') phase 2: ramp 5 at lr $best (the best of phase 1)" >> "$LOG"
run --lr "$best" --stage-ramp-epochs 5 || exit 0
echo "$(date '+%F %T') layer-wise at 4/4 finished" >> "$LOG"
