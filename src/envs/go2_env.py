import gymnasium as gym
import numpy as np
import mujoco

class Go2Env(gym.Env):
    def __init__(self, xml_path: str = "third_party/mujoco_menagerie/unitree_go2/scene.xml", terrain_mode: str = "flat"):
        super().__init__()
        
        # 1. Load standard MuJoCo Model[cite: 1]
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. Modify Geometry and Friction Properties (Cylinder -> Capsule)
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
                
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        
        # Initialize Data object[cite: 1]
        self.data = mujoco.MjData(self.model)
        
        # Control frequency: 5 substeps (matches n_frames=5 in JAX)
        self.n_frames = 5
        self.dt = self.model.opt.timestep * self.n_frames
        
        # 3. Define Gymnasium Spaces (48D Obs, 12D Action)[cite: 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        
        # Updated nominal posture (FL, FR, RL, RR)
        self.q_nom = np.array([
            0.0, 0.9, -1.8,  # Front Left
            0.0, 0.9, -1.8,  # Front Right
            0.0, 0.9, -1.8,  # Rear Left
            0.0, 0.9, -1.8   # Rear Right
        ], dtype=np.float32)
        
        self.action_scale = 0.5  
        self.target_height = 0.28 

        self.last_action = np.zeros(12, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    def _quat_rotate_inverse(self, q, v):
        """Rotates a world-frame vector into the local body frame using the inverse quaternion."""
        w, x, y, z = q[0], -q[1], -q[2], -q[3]
        q_vec = np.array([x, y, z], dtype=np.float32)
        uv = np.cross(q_vec, v)
        uuv = np.cross(q_vec, uv)
        return v + 2.0 * (w * uv + uuv)

    def _get_obs(self) -> np.ndarray:
        quat = self.data.qpos[3:7].copy()
        
        # Transform velocities to local frame
        v_world = self.data.qvel[:3].copy()
        omega_world = self.data.qvel[3:6].copy()
        v_local = self._quat_rotate_inverse(quat, v_world)
        omega_local = self._quat_rotate_inverse(quat, omega_world)
        
        # Gravity projection
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        g_proj = self._quat_rotate_inverse(quat, g_world)
        
        q_joints = self.data.qpos[7:19].copy()
        dq_joints = self.data.qvel[6:18].copy()
        
        obs = np.concatenate([
            v_local,
            omega_local,
            g_proj,
            q_joints - self.q_nom,
            dq_joints * 0.05,
            self.last_action,
            self.commands
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply initial posture noise
        qpos = self.model.qpos0.copy()
        qpos[2] = self.target_height
        qpos[7:19] = self.q_nom + self.np_random.uniform(-0.05, 0.05, size=12)
        
        qvel = np.zeros(self.model.nv, dtype=np.float32)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        
        # --- ADVANCED COMMAND SAMPLING LOGIC ---
        speed = self.np_random.uniform(0.0, 1.2)
        angle = self.np_random.uniform(-0.5, 0.5)
        
        v_x = speed * np.cos(angle)
        v_y = speed * np.sin(angle)
        
        raw_yaw = self.np_random.uniform(-1.0, 1.0)
        speed_penalty = (speed / 1.2) * 0.85
        omega_z = raw_yaw * (1.0 - speed_penalty)
        
        self.commands = np.array([v_x, v_y, omega_z], dtype=np.float32)
        
        info = {
            "is_crashed": False,
            "commands": self.commands.copy()
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        # Action Scaling
        target_angles = self.q_nom + (action * self.action_scale)
        
        # Standard CPU MuJoCo Control Loop[cite: 1]
        self.data.ctrl[:] = target_angles
        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)
            
        # Kinematics & States
        quat = self.data.qpos[3:7].copy()
        v_world = self.data.qvel[:3].copy()
        omega_world = self.data.qvel[3:6].copy()
        
        v_local = self._quat_rotate_inverse(quat, v_world)
        omega_local = self._quat_rotate_inverse(quat, omega_world)
        
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        g_proj = self._quat_rotate_inverse(quat, g_world)
        
        base_z = self.data.qpos[2]
        q_joints = self.data.qpos[7:19].copy()
        dq_joints = self.data.qvel[6:18].copy()
        
        # --- SENSOR EXTRACTION ---
        # Assuming identical sensor layout to the JAX mjcf definition
        foot_forces = self.data.sensordata[-9:-5]
        num_feet_touching = np.sum(foot_forces > 0.1)
        
        has_illegal_touch = np.any(self.data.sensordata[-5:] > 0.1)
        
        # --- TERMINATION LOGIC ---
        is_inverted = bool(g_proj[2] > -0.4)
        is_crashed = bool(is_inverted or has_illegal_touch)
        
        reward = self._calc_reward(
            v_local, omega_local, g_proj, base_z,
            q_joints, dq_joints, action, is_crashed, num_feet_touching
        )
        
        self.last_action = action.copy()
        self.step_count += 1
        obs = self._get_obs()
        
        info = {
            "is_crashed": is_crashed,
            "commands": self.commands.copy()
        }
        
        # Terminated handles crashes, Truncated is managed by Gym TimeLimit wrapper[cite: 1]
        return obs, reward, is_crashed, False, info

    def _calc_reward(
        self, v_local, omega_local, g_proj, base_z,
        q_joints, dq_joints, action, is_crashed, num_feet_touching
    ):
        lin_vel_error = np.sum(np.square(v_local[:2] - self.commands[:2]))
        r_lin_vel = np.exp(-lin_vel_error / 0.25) * 1.5
        
        ang_vel_error = np.square(omega_local[2] - self.commands[2])
        r_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.8
        
        r_z_vel = -np.square(v_local[2]) * 1.0
        r_ang_rates = -np.sum(np.square(omega_local[:2])) * 0.05
        r_flat_posture = -np.sum(np.square(g_proj[:2])) * 2.5
        r_height = -np.square(base_z - self.target_height) * 10.0
        r_action_rate = -np.sum(np.square(action - self.last_action)) * 0.02
        r_joint_vel = -np.sum(np.square(dq_joints)) * 0.0001
        r_joint_nominal = -np.sum(np.square(q_joints - self.q_nom)) * 0.02
        
        r_airborne = -0.2 if num_feet_touching == 0 else 0.0
        r_alive = -1.0 if is_crashed else 0.5
        
        total_reward = (
            r_lin_vel + r_ang_vel + r_z_vel + r_ang_rates + r_flat_posture + 
            r_height + r_action_rate + r_joint_vel + r_joint_nominal + 
            r_airborne + r_alive
        )
        
        return float(np.clip(total_reward, -5.0, 10.0) * 0.02)