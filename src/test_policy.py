import os
import time
import threading
import tkinter as tk
from tkinter import ttk

import torch
import mujoco
import mujoco.viewer
import numpy as np

from src.networks.actor_critic import Actor

# --- Shared Command State ---
class InteractiveCommandState:
    def __init__(self):
        self.v_x = 0.0
        self.v_y = 0.0
        self.omega_z = 0.0
        self.should_reset = False
        self.running = True

    def get_command_array(self) -> np.ndarray:
        return np.array([self.v_x, self.v_y, self.omega_z], dtype=np.float32)


# --- Tkinter Control Dashboard ---
def start_control_panel(cmd_state: InteractiveCommandState):
    root = tk.Tk()
    root.title("Unitree Go2 Teleop Controller")
    root.geometry("380x300")
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use("clam")

    ttk.Label(root, text="Go2 Policy Controller", font=("Helvetica", 14, "bold")).pack(pady=10)

    # Forward / Backward Slider (Vx)
    ttk.Label(root, text="Linear Velocity Vx (m/s):").pack()
    vx_slider = ttk.Scale(root, from_=-1.5, to=1.5, orient="horizontal", length=300)
    vx_slider.set(0.0)
    vx_slider.pack(pady=2)

    # Lateral Slider (Vy)
    ttk.Label(root, text="Lateral Velocity Vy (m/s):").pack()
    vy_slider = ttk.Scale(root, from_=-1.0, to=1.0, orient="horizontal", length=300)
    vy_slider.set(0.0)
    vy_slider.pack(pady=2)

    # Yaw Slider (Wz)
    ttk.Label(root, text="Angular Velocity Wz (rad/s):").pack()
    wz_slider = ttk.Scale(root, from_=-1.5, to=1.5, orient="horizontal", length=300)
    wz_slider.set(0.0)
    wz_slider.pack(pady=2)

    def update_values():
        cmd_state.v_x = float(vx_slider.get())
        cmd_state.v_y = float(vy_slider.get())
        cmd_state.omega_z = float(wz_slider.get())
        if cmd_state.running:
            root.after(20, update_values)

    def zero_all():
        vx_slider.set(0.0)
        vy_slider.set(0.0)
        wz_slider.set(0.0)

    def trigger_reset():
        cmd_state.should_reset = True

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=15)
    ttk.Button(btn_frame, text="Stop / Zero", command=zero_all).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Reset Robot", command=trigger_reset).grid(row=0, column=1, padx=5)

    def on_close():
        cmd_state.running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_values()
    root.mainloop()


# --- Math & Quaternion Helpers ---
def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates vector v from world frame to body frame using inverse quaternion [w, x, y, z]."""
    w, x, y, z = q[0], -q[1], -q[2], -q[3]
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (w * uv + uuv)


# --- Policy Checkpoint Loader ---
def load_trained_actor(ckpt_path: str, obs_dim: int = 48, action_dim: int = 12):
    actor = Actor(obs_dim=obs_dim, action_dim=action_dim)
    
    # Load on CPU to avoid unnecessary GPU VRAM usage during viewer test
    device = torch.device('cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract just the actor weights from the saved dictionary
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # Set network to evaluation mode (disables dropout/batchnorm if any existed)
    
    return actor


# --- Main Simulation Loop ---
def main():
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    # Make sure to point this to a generated .pt file!
    ckpt_path = os.path.abspath("./checkpoints/step_150.pt")  

    # Nominal joint configuration (Matching Go2Env)
    q_nom = np.array([
         0.1,  0.8, -1.5,  # Front Right
        -0.1,  0.8, -1.5,  # Front Left
         0.1,  1.0, -1.5,  # Rear Right
        -0.1,  1.0, -1.5   # Rear Left
    ])

    # 1. Load MuJoCo Model & Physics
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # 2. Load Neural Policy
    print(f"Loading checkpoint from: {ckpt_path}")
    # Note: obs_dim is now 48 (since we removed the 1D sitting command)
    actor = load_trained_actor(ckpt_path, obs_dim=48, action_dim=12)

    # 3. Launch UI Thread
    cmd_state = InteractiveCommandState()
    gui_thread = threading.Thread(target=start_control_panel, args=(cmd_state,), daemon=True)
    gui_thread.start()

    def reset_sim():
        mujoco.mj_resetData(m, d)
        d.qpos[7:19] = q_nom
        mujoco.mj_forward(m, d)

    reset_sim()
    last_action = np.zeros(12, dtype=np.float32)
    n_frames = 10  # 10 substeps @ 50 Hz control frequency

    print("Launching MuJoCo Viewer... Use the UI sliders to command the robot.")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and cmd_state.running:
            step_start = time.time()

            if cmd_state.should_reset:
                reset_sim()
                last_action = np.zeros(12, dtype=np.float32)
                cmd_state.should_reset = False

            # --- 1. Construct 48D Observation ---
            v_base = d.qvel[:3].copy()
            omega_base = d.qvel[3:6].copy()
            quat = d.qpos[3:7].copy()
            g_proj = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            q_joints = d.qpos[7:19].copy()
            dq_joints = d.qvel[6:18].copy()
            commands = cmd_state.get_command_array()

            obs = np.concatenate([
                v_base,
                omega_base,
                g_proj,
                q_joints - q_nom,
                dq_joints,
                last_action,
                commands
            ]).astype(np.float32)

            # --- 2. Policy Inference (Deterministic Mean) ---
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                # The PyTorch Actor forward pass returns (mean, log_std)
                mean_action, _ = actor(obs_tensor)
            
            action = mean_action.squeeze(0).numpy()
            action_clipped = np.clip(action, -1.0, 1.0)

            # --- 3. Action Scaling & PD Actuation ---
            # Adjusted back to 0.5 to match the new environment spec!
            action_scaled = action_clipped * 0.5
            target_angles = q_nom + action_scaled
            d.ctrl[:] = target_angles
            last_action = action_clipped

            # --- 4. Step Physics Engine ---
            for _ in range(n_frames):
                mujoco.mj_step(m, d)

            viewer.sync()

            # Maintain Real-Time Speed (50 Hz control loop = 0.02s per step)
            elapsed = time.time() - step_start
            sleep_time = (m.opt.timestep * n_frames) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()