import os
import time
import signal
import numpy as np
import torch
import gymnasium as gym

from src.envs.go2_env import Go2Env
from src.training.train_ppo import PPOConfig, create_train_states, ppo_train_iteration

def force_exit_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Forcing clean exit...")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit_handler)

def make_env(xml_path):
    """Factory function required by Gymnasium to spawn parallel processes."""
    def thunk():
        env = Go2Env(xml_path=xml_path)
        # Replaces Brax EpisodeWrapper: Limits episodes to 1000 steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env
    return thunk

def main():
    print("Initializing Unitree Go2 PyTorch CPU Training Pipeline...")
    
    # 1. Setup Configuration 
    # Perfectly sized for an 8-core CPU to avoid thread-thrashing overhead
    cfg = PPOConfig(
        num_envs=8,          
        num_steps=128,
        num_epochs=5,
        num_minibatches=8,
        entropy_coef=0.002,
        lr_actor=1e-4,
        lr_critic=5e-4,
    )
    
    # Force execution on the CPU
    device = torch.device("cpu")

    # 2. Instantiate Vectorized Environment
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    print(f"Spawning {cfg.num_envs} independent C++ MuJoCo physics threads...")
    
    # Replaces VmapWrapper & AutoResetWrapper (AsyncVectorEnv auto-resets by default)
    envs = gym.vector.AsyncVectorEnv([make_env(xml_path) for _ in range(cfg.num_envs)])
    
    # 3. Initialize State
    print("Initializing networks and optimizers...")
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    actor, critic, actor_optim, critic_optim = create_train_states(
        obs_dim, action_dim, cfg, device
    )
    
    current_obs, _ = envs.reset()
    current_done = np.zeros(cfg.num_envs, dtype=np.float32)

    # --- UPGRADE: PyTorch Checkpoint Resume Infrastructure ---
    ckpt_dir = os.path.abspath("./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_iteration = 0
    
    # Scan for standard PyTorch .pt files instead of Orbax directories
    existing_ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('step_') and d.endswith('.pt')]
    if existing_ckpts:
        ckpt_iters = [int(f.split('_')[1].split('.')[0]) for f in existing_ckpts]
        latest_step = max(ckpt_iters)
        print(f"[*] Found existing PyTorch checkpoint! Resuming from Iteration {latest_step}...")
        
        # Load the dictionary of weights
        checkpoint = torch.load(os.path.join(ckpt_dir, f"step_{latest_step}.pt"), map_location=device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        start_iteration = latest_step + 1

    total_iterations = 1000
    print("Starting Training Loop!")
    print("-" * 80)
    
    global_start_time = time.time()
    
    for i in range(start_iteration, total_iterations):
        start_time = time.time()
        
        # Execute the PyTorch monolithic training step
        current_obs, current_done, metrics = ppo_train_iteration(
            envs, current_obs, current_done, 
            actor, critic, actor_optim, critic_optim, 
            cfg, device
        )
        
        step_time = time.time() - start_time
        total_elapsed = time.time() - global_start_time
        
        # 4. Metric Tracking
        p_loss = metrics["policy_loss"]
        v_loss = metrics["value_loss"]
        entropy = metrics["entropy"]
        mean_reward = metrics["mean_reward"]
        crashes = metrics["total_crashes"]
        
        fps = (cfg.num_envs * cfg.num_steps) / step_time
        mins, secs = divmod(int(total_elapsed), 60)
        
        print(f"Iter {i:04d} | Time: {mins:02d}:{secs:02d} | FPS: {fps:5.0f} | Rew: {mean_reward:6.2f} | Falls: {crashes:4d} | P_Loss: {p_loss: .3f} | V_Loss: {v_loss: .3f} | Ent: {entropy: .3f}")
        
        # Save every 50 iterations
        if i % 50 == 0 and i > 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{i}.pt")
            # Save the full training state so AdamW momentum isn't lost on resume
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optim_state_dict': actor_optim.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
            }, ckpt_path)
            print(f"--> Saved checkpoint at iteration {i}")

if __name__ == "__main__":
    main()