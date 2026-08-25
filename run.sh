#!/bin/bash

# Ensure Python recognizes the root directory to import 'src'
export PYTHONPATH="$(pwd)"

# PyTorch CPU Multithreading Optimization:
# Restricts PyTorch to 1 thread per parallel environment to prevent CPU thrashing
export OMP_NUM_THREADS=1

# python -m src.training.train_ppo "$@"
# python -m src.tests.test_visual_reset

# Launch the PyTorch training loop
python -m src.main