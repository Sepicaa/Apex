#!/bin/bash
export PYTHONPATH="$(pwd)/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# python -m training.train_ppo "$@"
# python -m tests.test_visual_reset
python -m src.main