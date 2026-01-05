"""
train_ppo.py

Trains a PPO agent on the simulated world environment.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environment.gym_wrapper import SimulatedWorldEnv


def main():
    # ---------------------------
    # Training configuration
    # ---------------------------
    TOTAL_TIMESTEPS = 200_000
    ENV_KWARGS = {
        "grid_size": 10,
        "max_steps": 200,
        "seed": 42,
    }

    MODEL_DIR = "models/trained"
    LOG_DIR = "results/logs"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---------------------------
    # Vectorized environment
    # ---------------------------
    env = make_vec_env(
        SimulatedWorldEnv,
        n_envs=4,
        env_kwargs=ENV_KWARGS,
    )

    # ---------------------------
    # PPO model
    # ---------------------------
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # ---------------------------
    # Checkpointing
    # ---------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_sim_world",
    )

    # ---------------------------
    # Train
    # ---------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
    )

    # ---------------------------
    # Save final model
    # ---------------------------
    model.save(os.path.join(MODEL_DIR, "ppo_final"))

    env.close()
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()
