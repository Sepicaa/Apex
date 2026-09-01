import os
import time
import threading
import tkinter as tk
from tkinter import ttk

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import orbax.checkpoint as ocp

from src.training.networks import Actor


import signal
def force_exit_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Forcing clean exit...")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit_handler)
# --- Shared Command State ---
class InteractiveCommandState:
    def __init__(self):
        self.v_x = 0.0
        self.v_y = 0.0
        self.omega_z = 0.0
        self.should_reset = False
        self.running = True
        
        # Add metric trackers
        self.current_z_height = 0.0
        self.is_crashed = False

    def get_command_array(self) -> np.ndarray:
        return np.array([self.v_x, self.v_y, self.omega_z], dtype=np.float32)


# --- Tkinter Control Dashboard ---
def start_control_panel(cmd_state: InteractiveCommandState):
    root = tk.Tk()
    root.title("Unitree Go2 Teleop Controller")
    root.geometry("400x380") # Made slightly taller for the metrics
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use("clam")

    ttk.Label(root, text="Go2 Policy Controller", font=("Helvetica", 14, "bold")).pack(pady=10)

    # --- Live Metrics Frame ---
    metrics_frame = ttk.LabelFrame(root, text="Live Robot Metrics")
    metrics_frame.pack(fill="x", padx=20, pady=5)
    
    z_label = ttk.Label(metrics_frame, text="Base Z-Height: 0.000 m", font=("Courier", 10))
    z_label.pack(anchor="w", padx=10, pady=2)
    
    status_label = ttk.Label(metrics_frame, text="Status: ALIVE", font=("Courier", 10, "bold"), foreground="green")
    status_label.pack(anchor="w", padx=10, pady=2)
    
    vx_slider = ttk.Scale(root, from_=-1.5, to=1.5, orient="horizontal", length=300)
    vx_slider.set(0.0)
    vx_slider.pack(pady=2)

    ttk.Label(root, text="Lateral Velocity Vy (m/s):").pack()
    vy_slider = ttk.Scale(root, from_=-1.0, to=1.0, orient="horizontal", length=300)
    vy_slider.set(0.0)
    vy_slider.pack(pady=2)

    ttk.Label(root, text="Angular Velocity Wz (rad/s):").pack()
    wz_slider = ttk.Scale(root, from_=-1.5, to=1.5, orient="horizontal", length=300)
    wz_slider.set(0.0)
    wz_slider.pack(pady=2)

    def update_values():
        cmd_state.v_x = float(vx_slider.get())
        cmd_state.v_y = float(vy_slider.get())
        cmd_state.omega_z = float(wz_slider.get())
        
        # Update Live Metrics UI
        z_label.config(text=f"Base Z-Height: {cmd_state.current_z_height:.3f} m")
        if cmd_state.is_crashed:
            status_label.config(text="Status: CRASHED", foreground="red")
        else:
            status_label.config(text="Status: ALIVE", foreground="green")
            
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
def load_trained_actor(ckpt_dir: str, obs_dim: int = 48, action_dim: int = 12):
    actor = Actor(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    key = jax.random.PRNGKey(0)
    empty_params = actor.init(key, dummy_obs)

    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_dir, target=empty_params)
    return actor, restored_params


# --- Main Simulation Loop ---
def main(num):
    # Use the mjx compiled XML so sensors perfectly match what the policy was trained on
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene_mjx.xml"
    ckpt_path = os.path.abspath(f"./checkpoints/step_{num}")  # Change to your target checkpoint

    q_nom = np.array([
        0.0, 0.9, -1.8,  # Front Left
        0.0, 0.9, -1.8,  # Front Right
        0.0, 0.9, -1.8,  # Rear Left
        0.0, 0.9, -1.8   # Rear Right
    ])

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print(f"Loading checkpoint from: {ckpt_path}")
    actor, params = load_trained_actor(ckpt_path, obs_dim=48, action_dim=12)

    cmd_state = InteractiveCommandState()
    gui_thread = threading.Thread(target=start_control_panel, args=(cmd_state,), daemon=True)
    gui_thread.start()

    def reset_sim():
        mujoco.mj_resetData(m, d)
        
        # 1. Randomize Base Z-Height (e.g., between 0.25m and 0.35m)
        d.qpos[2] = np.random.uniform(0.25, 0.35)
        
        # 2. Randomize Joint Angles
        # Bounded between -0.15 and 0.15 radians to maintain physical viability
        joint_noise = np.random.uniform(-0.15, 0.15, size=12)
        d.qpos[7:19] = q_nom + joint_noise
        
        # 3. Randomize Base Heading (Yaw)
        # Calculates a valid quaternion [w, x, y, z] for a random Z-axis rotation
        yaw = np.random.uniform(-np.pi, np.pi)
        d.qpos[3:7] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        
        # 4. Optional: Add a slight initial velocity push to test recovery
        d.qvel[0:2] = np.random.uniform(-0.2, 0.2, size=2) 
        
        mujoco.mj_forward(m, d)

    reset_sim()
    last_action = np.zeros(12, dtype=np.float32)
    n_frames = 5  # 10 substeps @ 50 Hz control frequency

    print("Launching MuJoCo Viewer... Use the UI sliders to command the robot.")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Track the base body
        base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance = 2.5 
        viewer.cam.elevation = -25 
        viewer.cam.azimuth = 135

        # --- NEW CODE: Force WSL Performance Settings ---
        # 1. Disable Heavy OpenGL Effects (Matches the bottom-left of your UI)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0

        # 2. Disable unnecessary Model Elements (Matches the top-left of your UI)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        # Ensure the ground plane stays visible!
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1 
        
        # Apply the changes to the viewer before the while loop starts
        viewer.sync()
        
        # 2. Performance tweaks: Safely update available flags if needed, 
        # or just omit them since modern mujoco-python handles rendering efficiently.
        # try:
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
        # except KeyError:
        #     pass
        
        render_fps = 30
        render_interval = 1.0 / render_fps
        last_render_time = time.time()

        while viewer.is_running() and cmd_state.running:
            step_start = time.time()
            step_start = time.time()

            if cmd_state.should_reset:
                reset_sim()
                last_action = np.zeros(12, dtype=np.float32)
                cmd_state.should_reset = False


            # --- Update Tkinter Dashboard Metrics ---
            cmd_state.current_z_height = d.qpos[2]
            
            # Check for crashes using the belly/thigh sensors (index -5 to end)
            has_illegal_touch = np.any(d.sensordata[-5:] > 0.1)
            is_inverted = quat_rotate_inverse(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))[2] > -0.4
            cmd_state.is_crashed = bool(has_illegal_touch or is_inverted)

            # --- 1. Construct 48D Observation (Mirroring Go2Env exactly) ---
            quat = d.qpos[3:7]  # [w, x, y, z]
            
            # Local velocities
            v_local = quat_rotate_inverse(quat, d.qvel[:3])
            omega_local = quat_rotate_inverse(quat, d.qvel[3:6])
            g_proj = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            
            q_joints = d.qpos[7:19]
            dq_joints = d.qvel[6:18]
            commands = cmd_state.get_command_array()

            obs = np.concatenate([
                v_local,
                omega_local,
                g_proj,
                q_joints - q_nom,
                dq_joints * 0.05,
                last_action,
                commands
            ]).astype(np.float32)

            # Armor the network against simulation physics spikes
            obs = np.clip(obs, -10.0, 10.0)

            # --- 2. Policy Inference ---
            obs_jax = jnp.array(obs)[None, :]
            mean_action, _ = actor.apply(params, obs_jax)
            action = np.array(mean_action[0])
            action_clipped = np.clip(action, -1.0, 1.0)

            # --- 3. Action Scaling & Actuation ---
            action_scaled = action_clipped * 0.5
            target_angles = q_nom + action_scaled
            d.ctrl[:] = target_angles
            last_action = action_clipped

            # --- 4. Step Physics Engine ---
            for _ in range(n_frames):
                    mujoco.mj_step(m, d)

            # ONLY render if 1/30th of a second has passed
            if time.time() - last_render_time >= render_interval:
                viewer.sync()
                last_render_time = time.time()

            # Sleep to maintain real-time physics pacing
            elapsed = time.time() - step_start
            sleep_time = (m.opt.timestep * n_frames) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(240)
        print("add the step number")