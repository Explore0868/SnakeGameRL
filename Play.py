
import sys
import time
from stable_baselines3 import DQN

from snake_py import SnakeEnv , Gridsize

GRID_W ,GRID_H = Gridsize.assign_grid_size(self=True)

#Or else we can directly take from Config Module

import Config as con_grid

GRID_H = con_grid.GRID_H
GRID_W = con_grid.GRID_W

DEFAULT_MODEL_PATH = "models/best_model.zip"
# Grid size now lives in config.py, shared with train.py -- always matches or directly from SnakeEnv Module
EPISODES = 1


def load_model(path):
    
    try:
        return DQN.load(path)
    except Exception:
        raise ValueError(f"Could not load model at {path} as DQN ")


def main():

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    env = SnakeEnv(grid_w=GRID_W, grid_h=GRID_H, render_mode="human")

    scores = []
    for ep in range(1, EPISODES + 1):
        obs, info = env.reset()
        done = False
        while not done:
            # deterministic=True -> always pick the best known action,
            # no exploration randomness (that's only for training)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            env.render()

        scores.append(info["score"])
        print(f"Episode {ep}: score = {info['score']}")
        time.sleep(0.5)

    print(f"\nAverage score over {EPISODES} episodes: {sum(scores) / len(scores):.2f}")
    env.close()


if __name__ == "__main__":
    main()