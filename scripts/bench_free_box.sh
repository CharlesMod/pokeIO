#!/bin/bash
cd /home/cmod/pokeIO
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/cmod/pokeIO
rm -rf runs/perf32 runs/perf64
echo "=== P32 clean ==="
python -m pokeio.train --gens 3 --pop 64 --players 32 --episode-steps 400 --goexplore --run-id perf32 --no-live
echo "=== P64 clean ==="
python -m pokeio.train --gens 3 --pop 64 --players 64 --episode-steps 400 --goexplore --run-id perf64 --no-live
echo "=== DONE ==="
