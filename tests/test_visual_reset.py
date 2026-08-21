import jax
import jax.numpy as jnp
from brax.io import html
from envs.go2_env import Go2Env

def main():
    print("--- 1. Initializing Environment ---")
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    env = Go2Env(xml_path=xml_path)

    print("--- 2. Compiling and Executing Reset ---")
    rng = jax.random.PRNGKey(42)
    # JIT-compile reset to ensure XLA compatibility
    jit_reset = jax.jit(env.reset)
    state = jit_reset(rng)

    print("State successfully generated!")
    print(f"Observation shape: {state.obs.shape} (Expected: (48,))")
    print(f"Reward: {state.reward}")
    print(f"Done: {state.done}")
    print(f"Base Position (Z-height): {state.pipeline_state.qpos[2]:.3f} m")

    print("\n--- 3. Generating Interactive 3D HTML ---")
    # Brax takes a list of pipeline states (even just 1 frame) to render
    html_content = html.render(env.sys, [state.pipeline_state])
    
    output_path = "reset_scene.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Interactive 3D view saved to: {output_path}")

if __name__ == "__main__":
    main()