import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Tuple

from src.training.networks import Actor, Critic
from src.training.rollout import collect_rollouts, Transition
from src.training.gae import compute_gae
from src.envs.go2_env import Go2Env

class PPOConfig(NamedTuple):
    num_envs: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.05
    num_steps: int = 128
    num_epochs: int = 5
    num_minibatches: int = 8
    max_grad_norm: float = 0.5


# --- 1. Loss Functions ---

def actor_loss_fn(actor_params, actor: Actor, batch: Transition, advantages: jax.Array, clip_eps: float, entropy_coef: float):
    mean, log_std = actor.apply(actor_params, batch.obs)
    std = jnp.exp(log_std)

    log_prob = -0.5 * jnp.sum(
        ((batch.action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1
    )

    ratio = jnp.exp(log_prob - batch.log_prob)

    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    entropy = jnp.mean(jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2 * jnp.pi)), axis=-1))
    
    total_actor_loss = policy_loss - entropy_coef * entropy
    return total_actor_loss, (policy_loss, entropy)


def critic_loss_fn(critic_params, critic: Critic, batch_obs: jax.Array, targets: jax.Array):
    values = critic.apply(critic_params, batch_obs)
    value_loss = 0.5 * jnp.mean(jnp.square(values - targets))
    return value_loss, value_loss


# --- 2. Update Epoch on Minibatches ---

def update_minibatch(actor_state: TrainState, critic_state: TrainState,
                     actor: Actor, critic: Critic,
                     minibatch: Transition, mb_advantages: jax.Array, mb_targets: jax.Array,
                     cfg: PPOConfig):
    
    grad_actor_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (_, (p_loss, ent)), actor_grads = grad_actor_fn(
        actor_state.params, actor, minibatch, mb_advantages, cfg.clip_eps, cfg.entropy_coef
    )
    new_actor_state = actor_state.apply_gradients(grads=actor_grads)

    grad_critic_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    (_, v_loss), critic_grads = grad_critic_fn(
        critic_state.params, critic, minibatch.obs, mb_targets
    )
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)

    metrics = {
        "policy_loss": p_loss,
        "value_loss": v_loss,
        "entropy": ent
    }
    return new_actor_state, new_critic_state, metrics


# --- 3. Multi-Epoch Minibatch Update ---

def update_epoch(
    actor_state: TrainState, critic_state: TrainState,
    actor: Actor, critic: Critic,
    trajectories: Transition, advantages: jax.Array, targets: jax.Array,
    cfg: PPOConfig, rng: jax.Array
):
    batch_size = cfg.num_steps * cfg.num_envs
    minibatch_size = batch_size // cfg.num_minibatches

    flat_trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
    )
    flat_adv = advantages.reshape((batch_size,))
    flat_targets = targets.reshape((batch_size,))

    norm_advantages = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

    def _epoch_step(carry, _):
        act_state, crit_state, key = carry
        key, perm_key = jax.random.split(key)

        permutation = jax.random.permutation(perm_key, batch_size)
        
        shuffled_batch = jax.tree_util.tree_map(lambda x: x[permutation], flat_trajectories)
        shuffled_adv = norm_advantages[permutation]
        shuffled_targets = flat_targets[permutation]

        mb_batches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (cfg.num_minibatches, minibatch_size) + x.shape[1:]),
            shuffled_batch
        )
        mb_adv = jnp.reshape(shuffled_adv, (cfg.num_minibatches, minibatch_size))
        mb_targ = jnp.reshape(shuffled_targets, (cfg.num_minibatches, minibatch_size))

        def _minibatch_step(inner_carry, mb_data):
            a_st, c_st = inner_carry
            mb_trans, mb_a, mb_t = mb_data
            new_a_st, new_c_st, metrics = update_minibatch(
                a_st, c_st, actor, critic, mb_trans, mb_a, mb_t, cfg
            )
            return (new_a_st, new_c_st), metrics

        (act_state, crit_state), epoch_metrics = jax.lax.scan(
            _minibatch_step,
            (act_state, crit_state),
            (mb_batches, mb_adv, mb_targ)
        )

        return (act_state, crit_state, key), epoch_metrics

    initial_carry = (actor_state, critic_state, rng)
    (final_actor_state, final_critic_state, _), all_metrics = jax.lax.scan(
        _epoch_step, initial_carry, None, length=cfg.num_epochs
    )

    return final_actor_state, final_critic_state, all_metrics


# --- 4. TrainState Initializer Helper ---

def create_train_states(
    actor: Actor, critic: Critic,
    obs_dim: int, cfg: PPOConfig, rng: jax.Array
) -> Tuple[TrainState, TrainState]:
    rng_act, rng_crit = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))

    actor_params = actor.init(rng_act, dummy_obs)
    actor_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=cfg.lr_actor, weight_decay=cfg.weight_decay)
    )
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)

    critic_params = critic.init(rng_crit, dummy_obs)
    critic_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=cfg.lr_critic, weight_decay=cfg.weight_decay)
    )
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_tx)

    return actor_state, critic_state


# --- 5. Single PPO Iteration Step ---

def ppo_train_iteration(
    env: Go2Env, env_state,
    actor_state: TrainState, critic_state: TrainState,
    actor: Actor, critic: Critic,
    cfg: PPOConfig, rng: jax.Array, global_iteration: jax.Array
):
    rng_rollout, rng_update = jax.random.split(rng)

    (next_env_state, last_obs, _), trajectories = collect_rollouts(
        env, env_state, actor, critic,
        actor_state.params, critic_state.params,
        rng_rollout, cfg.num_steps, global_iteration
    )

    last_val = critic.apply(critic_state.params, last_obs)

    advantages, targets = compute_gae(
        trajectories, last_val, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
    )

    new_actor_state, new_critic_state, metrics = update_epoch(
        actor_state, critic_state, actor, critic,
        trajectories, advantages, targets, cfg, rng_update
    )
    
    # Export full trajectories to Python for rolling episodic tracking
    metrics["step_rewards"] = trajectories.reward
    metrics["step_dones"] = trajectories.done
    metrics["step_crashes"] = trajectories.is_crashed
    
    return next_env_state, new_actor_state, new_critic_state, metrics