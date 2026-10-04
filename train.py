"""
Training script for the Snake RL agent.

Switch algorithms by changing ALGORITHM below to "DQN" or "PPO".
Both come from stable-baselines3, so the rest of the code stays identical
either way -- that's the point of using a library instead of hand-coding
Q-learning update rules.
"""

import os
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from snake_py import SnakeEnv , Gridsize

GRID_W ,GRID_H = Gridsize.assign_grid_size(self=True)

#Or else we can directly take from Config Module

import Config as con_grid

GRID_H = con_grid.GRID_H
GRID_W = con_grid.GRID_W


# CONFIG -- change this one line to switch algorithms


ALGORITHM = "DQN"          
TOTAL_TIMESTEPS = 500_000  
# Grid size now lives in config.py, shared with play.py

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    return SnakeEnv(grid_w=GRID_W, grid_h=GRID_H, render_mode=None)


def build_model(algo_name, env):
    """
    Both DQN and PPO use a simple MLP ("Multi-Layer Perceptron" -- a plain
    feed-forward neural network) since our state is just 11 numbers, not
    an image. No convolutional layers needed.
    """
    # net_arch=[256, 256] gives the network more capacity than SB3's default
    # (distance shaping + space awareness), since a bigger network can
    
    policy_kwargs = dict(net_arch=[256, 256])

    if algo_name == "DQN":
        return DQN(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-3,
            buffer_size=100_000,       # replay buffer size (see explanation above)
            learning_starts=1_000,     # collect some random experience before training starts
            batch_size=64,
            gamma=0.99,                # discount factor
            train_freq=4,
            target_update_interval=1_000,  # how often the "target network" syncs
            exploration_fraction=0.2,      # % of training spent exploring more
            exploration_final_eps=0.02,    # minimum randomness kept even late in training
            verbose=1,
            tensorboard_log=LOG_DIR,
        )
    elif algo_name == "PPO":
        return PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )



def main():
    # vectorized env just means "wrap it so SB3 can manage resets automatically"
    # monitor_dir tells SB3 to log every episode's score/length to a CSV file
    env = make_vec_env(make_env, n_envs=1, monitor_dir=LOG_DIR)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = build_model(ALGORITHM, env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Training {ALGORITHM} for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    final_path = os.path.join(MODEL_DIR, f"{ALGORITHM.lower()}_snake_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")
    print(f"Best model (by eval score) saved to {MODEL_DIR}/best_model.zip")


if __name__ == "__main__":
    main()