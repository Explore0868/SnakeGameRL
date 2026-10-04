"""
 to see how the agent is improving.
python plot_progress.py
"""

import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

LOG_DIR = "logs"


def moving_average(values, window=50):
    if len(values) < window:
        return values
    weights = [1.0 / window] * window
    return [
        sum(values[i - window:i]) / window
        for i in range(window, len(values) + 1)
    ]


def main():
    results = load_results(LOG_DIR)
    x, y = ts2xy(results, "timesteps")  # x = timesteps, y = episode reward

    if len(y) == 0:
        print("No episodes logged yet. Let training run a bit longer.")
        return

    smoothed = moving_average(y, window=min(50, len(y)))
    smoothed_x = x[len(x) - len(smoothed):]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.3, label="raw episode reward")
    plt.plot(smoothed_x, smoothed, label="smoothed (50-episode avg)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("training_progress.png")
    plt.show()
    print("Saved plot to training_progress.png")


if __name__ == "__main__":
    main()