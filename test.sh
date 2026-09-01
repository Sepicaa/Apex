#!/bin/bash
export PYTHONPATH="$(pwd)/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# python -m training.train_ppo "$@"
export LIBGL_ALWAYS_SOFTWARE=0
export GALLIUM_DRIVER=d3d12
export MESA_GL_VERSION_OVERRIDE=4.6
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
# export XDG_SESSION_TYPE=x11
# export GDK_BACKEND=x11
# python -m tests.test_enviroment
python -m src.test_policy "$@"