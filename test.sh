#!/bin/bash
export PYTHONPATH="$(pwd)/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# python -m training.train_ppo "$@"
# python -m tests.test_visual_reset
export LIBGL_ALWAYS_SOFTWARE=0
export GALLIUM_DRIVER=d3d12
export MESA_GL_VERSION_OVERRIDE=4.6
python -m src.test_policy