#!/bin/bash

# Ensure root directory is on PYTHONPATH
export PYTHONPATH="$(pwd)"

# Disable aggressive GPU preallocation so JAX allocates memory dynamically without warnings
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run training orchestrator
python -m training.train_ppo "$@"