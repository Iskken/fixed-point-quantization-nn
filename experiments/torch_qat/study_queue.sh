#!/bin/bash
# Runs task files listed in results/torch_qat/study/queue.txt one after another (one GPU job at a time).
# Lines can be appended while it runs; a processed line is recorded in queue_done.txt; "STOP" ends it.
cd "$(dirname "$0")/../.."
Q=results/torch_qat/study/queue.txt; DONE=results/torch_qat/study/queue_done.txt
touch "$Q" "$DONE"
while true; do
    next=$(grep -vxF -f "$DONE" "$Q" | head -1)
    if [ -z "$next" ]; then sleep 30; continue; fi
    if [ "$next" = "STOP" ]; then echo "STOP" >> "$DONE"; exit 0; fi
    # wait for any study run already on the GPU
    while pgrep -f "^[^ ]*python[^ ]* -m experiments.torch_qat.run_study" > /dev/null; do sleep 15; done
    echo "$(date '+%F %T') start $next" >> results/torch_qat/study/logs/queue.log
    ~/miniconda3/envs/quant/bin/python -m experiments.torch_qat.run_study "results/torch_qat/study/tasks/$next.json" \
        > "results/torch_qat/study/logs/$next.log" 2>&1
    echo "$(date '+%F %T') end $next (exit $?)" >> results/torch_qat/study/logs/queue.log
    echo "$next" >> "$DONE"
done
