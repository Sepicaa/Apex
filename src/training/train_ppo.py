import jax
import jax.numpy as jnp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx, mjtGeom

class Go2Env(PipelineEnv):
    def __init__(self, xml_path: str, terrain_mode: str = "flat", **kwargs):
        # 1. Load the raw MuJoCo C-model
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. Patch CYLINDER to CAPSULE to prevent MJX compilation errors
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mjtGeom.mjGEOM_CYLINDER:
                mj_model.geom_type[i] = mjtGeom.mjGEOM_CAPSULE
                
        # 3. Pass the patched model to Brax
        sys = mjcf.load_model(mj_model)
        
        self.terrain_mode = terrain_mode
        
        # Nominal standing posture (rads)
        self.q_nom = jnp.array([
             0.1,  0.8, -1.5,  # Front Right
            -0.1,  0.8, -1.5,  # Front Left
             0.1,  1.0, -1.5,  # Rear Right
            -0.1,  1.0, -1.5   # Rear Left
        ])

        super().__init__(sys, backend='mjx', **kwargs)

    def _get_obs(self, data: mjx.Data, action: jax.Array, commands: jax.Array) -> jax.Array:
        """
        Constructs the 48-dimensional observation vector.
        Proprioception (45) + Joystick Commands (3)
        """
        # 1. Base Kinematics
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        
        # 2. Projected Gravity (Inner Ear)
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        # 3. Joint Kinematics
        q_joints = data.qpos[7:19]
        dq_joints = data.qvel[6:18]
        
        # 4. Concatenate all variables into a single flat array
        obs = jnp.concatenate([
            v_base,                 # 3D
            omega_base,             # 3D
            g_proj,                 # 3D
            q_joints - self.q_nom,  # 12D (Error from nominal posture)
            dq_joints,              # 12D
            action,                 # 12D (Action history smoothing)
            commands                # 3D [v_x, v_y, omega_z]
        ])
        
        return obs

    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        pass