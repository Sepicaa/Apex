import jax
import jax.numpy as jnp
from brax.io import html
from envs.go2_env import Go2Env

def main():
    print("--- 1. Initializing Environment ---")
    xml_path = "third_party/mujoco_menagerie/unitree_go2/scene.xml"
    env = Go2Env(xml_path=xml_path)

    # JIT compile the functions for GPU speed
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng)
    
    print("--- 2. Resetting Environment ---")
    state = jit_reset(rng_reset)
    
    pipeline_states = [state.pipeline_state]
    
    print("--- 3. Simulating 100 Random Steps ---")
    for i in range(500):
        # Split RNG for random action generation
        rng, rng_action = jax.random.split(rng)
        
        # Generate a random action array strictly between -1.0 and 1.0
        random_action = jax.random.uniform(rng_action, (12,), minval=-1.0, maxval=1.0)
        
        # Step the environment
        state = jit_step(state, random_action)
        
        # Save the physics state for the HTML visualizer
        pipeline_states.append(state.pipeline_state)
        
        # Stop early if the robot falls over
        if state.done:
            print(f"Robot collapsed at step {i}! Z-height: {state.pipeline_state.qpos[2]:.3f}m")
            break

    print("\n--- 4. Generating Interactive 3D HTML ---")
    html_content = html.render(env.sys, pipeline_states)
    
    output_path = "random_steps_scene.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Interactive 3D view saved to: {output_path}")

if __name__ == "__main__":
    main()