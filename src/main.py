import os
import time
import signal
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from src.envs.go2_env import Go2Env
from src.training.networks import Actor, Critic
from src.training.train_ppo import PPOConfig, create_train_states, ppo_train_iteration

def force_exit_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Forcing clean exit...")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit_handler)

def make_env(xml_path):
    """Factory function required by Gymnasium for parallel processes."""
    def thunk():
        env = Go2Env(xml_path=xml_path)
        # Replaces Brax EpisodeWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env
    return thunk

def main():
    print("Initializing Unitree Go2 PyTorch PPO Training Pipeline...")
    
    cfg = PPOConfig(
        num_envs=64,
        num_steps=64,
        num_epochs=5,
        num_minibatches=8,
        entropy_coef=0.002,
        lr_actor=5e-4,
        lr_critic=5e-4,
    )
    
    # Setup Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Replaces Brax VmapWrapper and AutoResetWrapper
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    envs = gym.vector.AsyncVectorEnv([make_env(xml_path) for _ in range(cfg.num_envs)])
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    print("Initializing networks and optimizers...")
    actor, critic, actor_optim, critic_optim = create_train_states(
        obs_dim, action_dim, cfg, device
    )
    
    current_obs, _ = envs.reset()
    current_done = np.zeros(cfg.num_envs, dtype=np.float32)

    # PyTorch Checkpoint Integration (Replaces Orbax)
    ckpt_dir = os.path.abspath("./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_iteration = 0
    
    existing_ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('step_') and d.endswith('.pt')]
    if existing_ckpts:
        ckpt_iters = [int(f.split('_')[1].split('.')[0]) for f in existing_ckpts]
        latest_step = max(ckpt_iters)
        print(f"[*] Found existing checkpoint! Resuming from Iteration {latest_step}...")
        
        checkpoint = torch.load(os.path.join(ckpt_dir, f"step_{latest_step}.pt"), map_location=device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        start_iteration = latest_step + 1

    total_iterations = 1000
    print("Starting Training Loop!")
    print("-" * 80)
    
    # Initialize trackers for the last 100 episodes
    ep_return_queue = deque(maxlen=100)
    ep_crash_queue = deque(maxlen=100)
    
    # Running tallies for the parallel environments
    running_returns = np.zeros(cfg.num_envs)
    running_steps = np.zeros(cfg.num_envs)

    global_start_time = time.time()
    
    for i in range(start_iteration, total_iterations):
        start_time = time.time()
        
        # PyTorch equivalent of compiled_train_step
        current_obs, current_done, metrics = ppo_train_iteration(
            envs, current_obs, current_done, 
            actor, critic, actor_optim, critic_optim, 
            cfg, device
        )
        
        step_time = time.time() - start_time
        total_elapsed = time.time() - global_start_time
        
        # 1. Extract step metrics (ensure ppo_train_iteration returns these as numpy arrays)
        step_rewards = metrics["step_rewards"]  
        step_dones = metrics["step_dones"]      
        
        # 2. Accumulate metrics step-by-step
        for t in range(cfg.num_steps):
            running_returns += step_rewards[t]
            running_steps += 1
            
            dones_at_t = step_dones[t]
            if np.any(dones_at_t):
                # Identify which specific environments finished
                finished_envs = dones_at_t > 0.5
                
                # A fall is any episode that terminates before the 1000-step timeout
                falls = (running_steps[finished_envs] < 1000).astype(float)
                
                # Push the completed episode totals to our rolling queues
                ep_return_queue.extend(running_returns[finished_envs].tolist())
                ep_crash_queue.extend(falls.tolist())
                
                # Reset the running tally for those specific environments
                running_returns[finished_envs] = 0.0
                running_steps[finished_envs] = 0.0
                
        # 3. Calculate rolling averages
        mean_reward = np.mean(ep_return_queue) if len(ep_return_queue) > 0 else 0.0
        mean_crashes = np.mean(ep_crash_queue) if len(ep_crash_queue) > 0 else 0.0
        
        p_loss = metrics["policy_loss"]
        v_loss = metrics["value_loss"]
        entropy = metrics["entropy"]
        
        fps = (cfg.num_envs * cfg.num_steps) / step_time
        mins, secs = divmod(int(total_elapsed), 60)
        
        print(f"Iter {i:04d} | Time: {mins:02d}:{secs:02d} | FPS: {fps:5.0f} | Rew: {mean_reward:6.2f} | Falls/Ep: {mean_crashes:4.2f} | P_Loss: {p_loss: .3f} | V_Loss: {v_loss: .3f} | Ent: {entropy: .3f}")
        
        # Save every 50 iterations
        if i % 50 == 0 and i > 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{i}.pt")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optim_state_dict': actor_optim.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
            }, ckpt_path)
            print(f"--> Saved checkpoint at iteration {i}")

if __name__ == "__main__":
    main()