#!/bin/bash
cd /home/cmod/pokeIO
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/cmod/pokeIO
echo "########## PROFILE epw=1 players=64 ##########"
POKEIO_PROF=1 python -m pokeio.train --gens 1 --pop 64 --players 64 --episode-steps 120 --run-id prof1 --no-live 2>&1 | grep -iE "prof|round|barrier|fwd|book|step_" | head -20
for epw in 4 8 16; do
  rm -rf runs/sw_$epw
  echo "########## epw=$epw players=64 ##########"
  python -m pokeio.train --gens 2 --pop 64 --players 64 --episode-steps 300 --envs-per-worker $epw --run-id sw_$epw --no-live 2>&1 | grep -iE "fleet up|worker procs" | head -2
  python3 -c "
import json
r=[json.loads(l)['data'] for l in open('runs/sw_$epw/telemetry.jsonl') if l.strip()]
print('  epw=$epw sps/gen=',[round(d['throughput_sps']) for d in r],'peak=',max(round(d['throughput_sps']) for d in r))
"
done
echo "=== SWEEP DONE ==="
