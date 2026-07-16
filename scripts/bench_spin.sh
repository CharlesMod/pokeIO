#!/bin/bash
cd /home/cmod/pokeIO
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/cmod/pokeIO
rm -rf runs/spin64 runs/spin112
python -m pokeio.train --gens 2 --pop 64 --players 64 --envs-per-worker 2 --episode-steps 300 --goexplore --run-id spin64 --no-live
python -m pokeio.train --gens 2 --pop 112 --players 112 --envs-per-worker 4 --episode-steps 300 --goexplore --run-id spin112 --no-live
echo "=== SPIN BENCH DONE ==="
