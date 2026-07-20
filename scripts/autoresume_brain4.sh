#!/bin/bash
# Auto-resume brain4 (+ dashboard) after a reboot.
#
# The box runs on solar power and has cycled several times mid-run; each time it
# killed the training process and the dashboard. This @reboot hook brings them back
# from the last checkpoint so training survives outages unattended. It uses the
# faithful --resume path (policy+optimizer+curriculum+iter) and --saccade-substeps 1
# to keep brain4 in the exact 2.49 Hz single-saccade regime it was trained in (no
# mid-run distribution shift). Self-limiting: exits once brain4 reaches 400 iters.
#
# To DISABLE: `crontab -l | grep -v autoresume_brain4 | crontab -`  (or edit crontab).
set -u
cd /home/cmod/pokeIO || exit 0
export PYTHONPATH=.
PY=.venv/bin/python
LOG=runs/_autoresume.log
RUN=brain4
ITERS=400

echo "[autoresume $(date)] boot hook; settling 30s" >> "$LOG"
sleep 30

maxit=$("$PY" - <<'PYEOF'
import json
mx = 0
try:
    for line in open('runs/brain4/brain_metrics.jsonl'):
        try: mx = max(mx, int(json.loads(line).get('iter', 0)))
        except Exception: pass
except Exception: pass
print(mx)
PYEOF
)
if [ "${maxit:-0}" -ge "$ITERS" ]; then
    echo "[autoresume $(date)] $RUN already complete (iter $maxit); nothing to do" >> "$LOG"
    exit 0
fi

# Dashboard (bracket trick avoids the pgrep self-match footgun).
if ! pgrep -f "[p]okeio.dash.serve" >/dev/null; then
    nohup "$PY" -m pokeio.dash.serve --port 8600 --host 0.0.0.0 >> runs/_dashserve.log 2>&1 &
    echo "[autoresume $(date)] started dashboard" >> "$LOG"
fi

# Training: resume from checkpoint in its own S=1 regime.
if ! pgrep -f "[b]rain_loop.*--run-id $RUN" >/dev/null; then
    nohup "$PY" -m pokeio.train.brain_loop \
        --n-envs 64 --iterations "$ITERS" --device cuda:1 --eval-every 10 \
        --checkpoint-every 10 --run-id "$RUN" --resume runs/$RUN/brain.pt \
        --saccade-substeps 1 --dashboard >> runs/_brain4.log 2>&1 &
    echo "[autoresume $(date)] resumed $RUN from checkpoint (was at iter $maxit)" >> "$LOG"
else
    echo "[autoresume $(date)] $RUN already running; skip" >> "$LOG"
fi
