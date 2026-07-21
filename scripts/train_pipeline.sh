#!/bin/bash
# Idempotent training-pipeline manager (survives the box's solar-power reboots).
#
# BRAIN6 era: FORWARD-PROGRESSION (vision-necessary) dual-GPU A/B, both arms
# CONCURRENT (independent guards, may launch in the same 5-min tick):
#   brain6r  reflex-only gaze,        cuda:0, --forward            -> to 400 iters
#   brain6g  LEARNED gaze (Path B),   cuda:1, --forward --learned-gaze -> to 400 iters
#
# WHY --forward: brain4/brain5 proved boot-reset episodes are solvable by an
# open-loop memorized script (mode collapse at progress 32.0), starving the
# learned gaze of gradient (battery verdict: learned head < random). --forward
# mixes spawn sources {boot, demo depth, Go-Explore archive frontier} with a
# self-tuning mass per source, so no single script solves the distribution and
# vision becomes necessary.
#
# CONCLUDED runs — never resumed here: brain4 (done at 400) and brain5
# (deliberately STOPPED at iter 210 = the approved brain6 warm-start seed).
#
# First launch of an arm: --warm-start runs/brain5/brain.pt (policy only,
# strict=False — the gaze keys drop cleanly into brain6r's gaze-less arch; fresh
# optimizer/curriculum/ent-coef/iter). Later ticks: --resume the arm's own
# checkpoint. NOTE --resume strict-loads, so brain6g must keep --learned-gaze on
# every launch (its checkpoints carry the gaze head) and brain6r must never gain
# it. Neither arm passes --saccade-substeps (derived S=2, 4.98 Hz, as brain5).
#
# Dashboards don't clobber: one serve.py on :8600 serves ALL runs; each trainer
# writes only its own runs/<id>/live.json (+ per-run control files). Pin tabs
# with /brain?run=brain6r and /brain?run=brain6g (no ?run= = newest by mtime).
#
# CPU: the arms run CONCURRENTLY, so each is pinned to its own NUMA node
# (--numa-node 0/1) — without this both fleets pin their spin-workers to the
# IDENTICAL deterministic core list (2-per-core on 32 cores, 24 idle). 28 envs
# per arm = one spin-worker per core of a 28-core node (2x Xeon E5-2690 v4).
#
# DISABLE: `crontab -l | grep -v train_pipeline | crontab -`
set -u
cd /home/cmod/pokeIO || exit 0
# Single-instance lock: a manual invocation racing a */5 cron tick in the same
# second launched DUPLICATE arms on 2026-07-20 (the pgrep guard can't see a child
# in the sub-second window before its cmdline exists). flock serializes runs;
# a concurrent instance exits immediately rather than double-launching.
exec 9>runs/_pipeline.lock
flock -n 9 || exit 0
export PYTHONPATH=.
PY=.venv/bin/python
LOG=runs/_pipeline.log
ITERS=400
SEED_CKPT=runs/brain5/brain.pt   # approved warm-start: brain5 iter 210

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

# STOP sentinel: `touch runs/_pipeline.stop` halts run management (dashboard keepalive
# above still runs); `rm runs/_pipeline.stop` resumes within 5 min from latest checkpoints.
[ -f runs/_pipeline.stop ] && exit 0

# arm <run-id> <device> [extra flags...]: drive one brain6 arm to $ITERS.
# Warm-start on first launch, resume-from-own-checkpoint afterwards. Guarded by
# the [b]racket pgrep + maxit, so repeated ticks are no-ops while it runs.
arm() {
    local id=$1 dev=$2 it src
    shift 2
    it=$(maxit "$id")
    [ "${it:-0}" -ge "$ITERS" ] && return 0
    running "$id" && return 0
    if [ -f "runs/$id/brain.pt" ]; then
        src="--resume runs/$id/brain.pt"
    else
        src="--warm-start $SEED_CKPT"
    fi
    nohup "$PY" -m pokeio.train.brain_loop --n-envs 28 --iterations "$ITERS" \
        --device "$dev" --forward --eval-every 10 --checkpoint-every 10 \
        --run-id "$id" --dashboard $src "$@" >> "runs/_$id.log" 2>&1 &
    echo "[pipeline $(date)] launched $id on $dev (was iter $it; src='$src')" >> "$LOG"
}

# Both arms every tick — separate GPUs, separate NUMA nodes, separate run dirs.
arm brain6r cuda:0 --numa-node 0
arm brain6g cuda:1 --learned-gaze --numa-node 1

if [ "$(maxit brain6r)" -ge "$ITERS" ] && [ "$(maxit brain6g)" -ge "$ITERS" ] \
    && ! running brain6r && ! running brain6g; then
    echo "[pipeline $(date)] brain6r + brain6g both at $ITERS; nothing to do" >> "$LOG"
fi
