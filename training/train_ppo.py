# training/train_ppo.py
import yaml
import jax
from envs.go2_env import Go2Env

def main():
    print(f"JAX Devices: {jax.devices()}")
    
    # 1. Load configurations
    with open("configs/phase1_ppo.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Initialize the environment
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    env = Go2Env(xml_path=xml_path, terrain_mode=config.get("terrain_mode", "flat"))
    
    print(f"Environment initialized. Action space: {env.action_size}")
    
    # 3. Setup Flax Networks and Optax Optimizers (Next Steps)
    # 4. Define the JIT-compiled PPO training loop

if __name__ == "__main__":
    main()