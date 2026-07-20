#!/bin/bash
# Idempotent training-pipeline manager (survives the box's solar-power reboots).
#
# Drives the run SEQUENCE and keeps the dashboard up, resuming from checkpoints:
#   brain4  reflex-only baseline, S=1 (2.49 Hz)         -> to 400 iters
#   brain5  Path-B LEARNED GAZE A/B, S=2 (4.98 Hz)       -> to 400 iters, --learned-gaze
#
# brain5 auto-launches only once brain4 has logged 400 iters AND its process has fully
# exited (so it has checkpointed/exported and freed cuda:1 — no GPU contention). Safe to
# call repeatedly: an @reboot hook recovers fast, a */5 cron drives the handoff + ongoing
# resume. Self-terminating once both runs reach 400 (just logs "nothing to do").
#
# DISABLE: `crontab -l | grep -v train_pipeline | crontab -`
set -u
cd /home/cmod/pokeIO || exit 0
export PYTHONPATH=.
PY=.venv/bin/python
LOG=runs/_pipeline.log
ITERS=400

maxit() {  # highest logged iter for run id $1 (0 if none)
    "$PY" - "$1" <<'PYEOF'
import sys, json
mx = 0
try:
    for line in open(f"runs/{sys.argv[1]}/brain_metrics.jsonl"):
        try: mx = max(mx, int(json.loads(line).get("iter", 0)))
        except Exception: pass
except Exception: pass
print(mx)
PYEOF
}
running() { pgrep -f "[b]rain_loop.*--run-id $1" >/dev/null; }  # bracket = no self-match

# Dashboard always up.
if ! pgrep -f "[p]okeio.dash.serve" >/dev/null; then
    nohup "$PY" -m pokeio.dash.serve --port 8600 --host 0.0.0.0 >> runs/_dashserve.log 2>&1 &
    echo "[pipeline $(date)] started dashboard" >> "$LOG"
fi

# Stage 1: brain4 (reflex-only baseline, S=1) to 400.
b4=$(maxit brain4)
if [ "${b4:-0}" -lt "$ITERS" ]; then
    if ! running brain4; then
        nohup "$PY" -m pokeio.train.brain_loop --n-envs 64 --iterations "$ITERS" \
            --device cuda:1 --eval-every 10 --checkpoint-every 10 --run-id brain4 \
            --resume runs/brain4/brain.pt --saccade-substeps 1 --dashboard \
            >> runs/_brain4.log 2>&1 &
        echo "[pipeline $(date)] resumed brain4 (was iter $b4)" >> "$LOG"
    fi
    exit 0
fi
# brain4 logged 400 but may still be exiting (final checkpoint/export/free GPU): wait.
if running brain4; then
    echo "[pipeline $(date)] brain4 hit $ITERS; letting it exit before brain5" >> "$LOG"
    exit 0
fi

# Stage 2: brain5 (Path-B LEARNED GAZE A/B, S=2) to 400 — cuda:1 is now free.
b5=$(maxit brain5)
if [ "${b5:-0}" -lt "$ITERS" ]; then
    if ! running brain5; then
        RESUME=""
        [ -f runs/brain5/brain.pt ] && RESUME="--resume runs/brain5/brain.pt"
        nohup "$PY" -m pokeio.train.brain_loop --n-envs 64 --iterations "$ITERS" \
            --device cuda:1 --eval-every 10 --checkpoint-every 10 --run-id brain5 \
            --learned-gaze $RESUME --dashboard >> runs/_brain5.log 2>&1 &
        echo "[pipeline $(date)] launched brain5 --learned-gaze (was iter $b5; resume='$RESUME')" >> "$LOG"
    fi
    exit 0
fi
echo "[pipeline $(date)] brain4 + brain5 both at $ITERS; nothing to do" >> "$LOG"
