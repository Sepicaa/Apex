# Project Apex: High-Dimensional Continuous Control Benchmark

This repository contains a hardware-accelerated reinforcement learning pipeline designed to benchmark continuous control and evolutionary algorithms on high-degree-of-freedom cyber-physical systems (like the Unitree Go2 Quadruped) using JAX, MJX, and Brax.

## Quick Start & Installation

Follow these steps to clone the repository, download the required physics assets, and set up the GPU-accelerated environment.

### Clone the Repository
Clone the project and fetch the DeepMind Quadruped models using a shallow clone to minimize download size:

```bash
git clone https://github.com/Sepicaa/Apex.git
cd Apex
git submodule update --init --depth 1
```
Next, initialize the environment and install the packages:

```Bash
python3 -m venv ai_venv
source ai_venv/bin/activate
pip install -U "jax[cuda12]"
pip install -r requirements.txt
```
### Usage
To ensure correct Python module routing and optimal GPU memory allocation, always launch the project from the root directory using a bash script.

Run the run.sh executable:
```Bash
chmod +x run.sh
./run.sh
```