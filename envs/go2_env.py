import jax
import jax.numpy as jnp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx, mjtGeom

class Go2Env(PipelineEnv):
    def __init__(self, xml_path: str, terrain_mode: str = "flat", **kwargs):
        # 1. Load the raw MuJoCo C-model first (do NOT call mjcf.load directly)
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 2. Patch CYLINDER -> CAPSULE before MJX conversion
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mjtGeom.mjGEOM_CYLINDER:
                mj_model.geom_type[i] = mjtGeom.mjGEOM_CAPSULE
                
        # 3. Pass the patched model to brax
        sys = mjcf.load_model(mj_model)
        
        self.terrain_mode = terrain_mode
        self.q_nom = jnp.array([
             0.1,  0.8, -1.5,
            -0.1,  0.8, -1.5,
             0.1,  1.0, -1.5,
            -0.1,  1.0, -1.5
        ])

        super().__init__(sys, backend='mjx', **kwargs)

    def _get_obs(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        v_base = data.qvel[:3]
        omega_base = data.qvel[3:6]
        
        quat = data.qpos[3:7]
        g_proj = math.rotate(jnp.array([0.0, 0.0, -1.0]), math.quat_inv(quat))
        
        q_joints = data.qpos[7:19]
        dq_joints = data.qvel[6:18]
        
        return jnp.concatenate([
            v_base,
            omega_base,
            g_proj,
            q_joints - self.q_nom,
            dq_joints,
            action
        ])

    def reset(self, rng: jax.Array) -> State:
        data = self.pipeline_init(self.sys.mj_model.qpos0, jnp.zeros(self.sys.nv))
        obs = self._get_obs(data, jnp.zeros(self.action_size))
        reward, done, zero = jnp.zeros(3)
        metrics = {}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        data = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(data, action)
        reward = 0.0
        done = 0.0
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)