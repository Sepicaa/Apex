import gymnasium as gym
import numpy as np
import mujoco

class Go2Env(gym.Env):
    def __init__(self, xml_path: str, terrain_mode: str = "flat"):
        super().__init__()
        
        # 1. Load standard MuJoCo C-Model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Control frequency: 50 Hz (10 frames * 0.002s timestep)
        self.n_frames = 10
        self.dt = self.model.opt.timestep * self.n_frames
        
        # 2. Define Gymnasium Spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        
        # Nominal standing posture
        self.q_nom = np.array([
             0.1,  0.8, -1.5,  # Front Right
            -0.1,  0.8, -1.5,  # Front Left
             0.1,  1.0, -1.5,  # Rear Right
            -0.1,  1.0, -1.5   # Rear Left
        ], dtype=np.float32)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)

    def _get_gravity_projection(self, quat):
        # Rotate [0, 0, -1] by the inverse of the base quaternion
        w, x, y, z = quat
        g_x = 2 * (x * z - w * y)
        g_y = 2 * (y * z + w * x)
        g_z = 1 - 2 * (x**2 + y**2)
        return np.array([-g_x, -g_y, -g_z], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        v_base = self.data.qvel[:3].copy()
        omega_base = self.data.qvel[3:6].copy()
        
        quat = self.data.qpos[3:7].copy()
        g_proj = self._get_gravity_projection(quat)
        
        q_joints = self.data.qpos[7:19].copy()
        dq_joints = self.data.qvel[6:18].copy()
        
        obs = np.concatenate([
            v_base,
            omega_base,
            g_proj,
            q_joints - self.q_nom,
            dq_joints,
            self.last_action,
            self.commands
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Add slight initial posture perturbation using NumPy RNG
        qpos = self.model.qpos0.copy()
        qpos[:7] += self.np_random.uniform(-0.01, 0.01, size=7)
        qpos[7:19] = self.q_nom + self.np_random.uniform(-0.05, 0.05, size=12)
        
        qvel = self.np_random.uniform(-0.01, 0.01, size=self.model.nv)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # Sample Locomotion Commands
        angle = self.np_random.uniform(-np.pi, np.pi)
        speed = self.np_random.uniform(0.0, 1.0)
        v_x = np.cos(angle) * speed
        v_y = np.sin(angle) * speed
        omega_z = self.np_random.uniform(-0.5, 0.5)
        self.commands = np.array([v_x, v_y, omega_z], dtype=np.float32)
        
        info = {"is_crashed": False}
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        # Action Scaling
        action_scaled = action * 0.5
        target_angles = self.q_nom + action_scaled
        
        # Standard CPU MuJoCo Control Loop
        self.data.ctrl[:] = target_angles
        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)
            
        # Kinematics
        v_base = self.data.qvel[:3].copy()
        omega_base = self.data.qvel[3:6].copy()
        z_height = self.data.qpos[2]
        quat = self.data.qpos[3:7].copy()
        g_proj = self._get_gravity_projection(quat)
        
        # Termination conditions
        is_flipped = g_proj[2] > 0.0
        is_bottomed_out = z_height < 0.20
        terminated = bool(is_flipped or is_bottomed_out)
        
        # Rewards (Integrated your Alive Bonus fix)
        reward = self._calc_reward(v_base, omega_base, action, g_proj, terminated)
        
        self.last_action = action.copy()
        obs = self._get_obs()
        
        info = {"is_crashed": terminated}
        
        # Truncated is False here; handled by Gymnasium TimeLimit wrapper later
        return obs, reward, terminated, False, info

    def _calc_reward(self, v_base, omega_base, action, g_proj, terminated):
        r_alive = 0.0 if terminated else 2.0
        
        linear_error = np.sum(np.square(v_base[:2] - self.commands[:2]))
        angular_error = np.square(omega_base[2] - self.commands[2])
        r_lin_track = np.exp(-linear_error / 0.25) * 1.0
        r_ang_track = np.exp(-angular_error / 0.25) * 1.0
        
        r_crash = -2.0 if terminated else 0.0
        r_flat = -np.sum(np.square(g_proj[:2])) * 5.0
        r_action_rate = -np.sum(np.square(action - self.last_action)) * 0.05
        
        return (r_alive + r_lin_track + r_ang_track + r_crash + r_flat + r_action_rate) * self.dt