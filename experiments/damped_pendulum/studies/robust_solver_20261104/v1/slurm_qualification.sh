#!/usr/bin/env bash
#SBATCH --job-name=robust-solver-qualification
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/weizhixin/ssd/robust_solver_20261104_v1/qualification-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/robust_solver_20261104_v1
ORIGINAL_ROOT=/home/weizhixin/ssd/dissipative_passive_20261103T051548Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$RUN_ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROBUST_TEST_DEVICE=cuda "$PYTHON_BIN" -m pytest tests/test_robust_dissipative_core.py -q > "$RUN_ROOT/cuda_tests.log"
"$PYTHON_BIN" scripts/profile_dissipative.py --data "$ORIGINAL_ROOT/data/passive_family_v1/seen.h5" --config configs/dissipative_passive_robust.json --output "$RUN_ROOT/fresh_profile.json" --device cuda --execute > "$RUN_ROOT/fresh_profile.log"
"$PYTHON_BIN" scripts/migrate_solver_checkpoint.py --checkpoint "$ORIGINAL_ROOT/runs/seed0_47541/last.pt" --config configs/dissipative_passive_robust.json --data "$ORIGINAL_ROOT/data/passive_family_v1/seen.h5" --original-source-root "$ORIGINAL_ROOT/code" --output "$RUN_ROOT/migration" --device cuda --write > "$RUN_ROOT/migration.log"
"$PYTHON_BIN" -m dpbench.train_dissipative --data "$ORIGINAL_ROOT/data/passive_family_v1/seen.h5" --config configs/dissipative_passive_robust.json --output "$RUN_ROOT/continuation" --device cuda --resume "$RUN_ROOT/migration/last.pt" --stop-after-step 4250 --execute > "$RUN_ROOT/continuation.log"
"$PYTHON_BIN" scripts/qualify_robust_solver.py --checkpoint "$RUN_ROOT/continuation/last.pt" --config configs/dissipative_passive_robust.json --data "$ORIGINAL_ROOT/data/passive_family_v1/seen.h5" --output "$RUN_ROOT/k32_stress" --device cuda --updates 500 --horizon 32 --rollout-steps 320 --execute > "$RUN_ROOT/k32_stress.log"
