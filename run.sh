#!/bin/bash
export PYTHONPATH="$(pwd)/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# python -m training.train_ppo "$@"
# python -m tests.test_random_steps
python -m src.main