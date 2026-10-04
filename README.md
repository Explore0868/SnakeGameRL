# Snake RL — Deep Q-Network Agent

A Snake-playing agent trained with Deep Q-Learning (DQN), using
[stable-baselines3](https://stable-baselines3.readthedocs.io/) for the RL
algorithm and Pygame for rendering.

## Overview

The agent learns to play Snake purely through trial and error — it is
never given rules like "avoid walls" or "seek food" directly. Instead, it
receives a numeric reward after every move and gradually learns a policy
(a strategy for choosing actions) that maximizes long-term reward.

| Component | Choice | Why |
|---|---|---|
| Algorithm | DQN (Deep Q-Network) | Standard, well-documented approach for discrete-action problems like Snake |
| Library | stable-baselines3 | Battle-tested implementation, avoids reimplementing core RL machinery |
| State representation | 14-value feature vector (not raw pixels) | Much faster to train than a CNN on raw grid images, while still capturing the information needed to play well |
| Action space | 3 relative actions (straight / turn right / turn left) | Easier to learn than 4 absolute directions (up/down/left/right), since it removes the "can't reverse into yourself" edge case entirely |

## Project Structure

```
snake-rl/
├── config.py           # Shared settings (grid size) used by train.py and play.py
├── snake_env.py         # Gymnasium environment: game logic, state, reward
├── train.py             # Trains the DQN agent with stable-baselines3
├── play.py              # Loads a trained model and plays it live in Pygame
├── plot_progress.py     # Plots training reward curve from logs
├── requirements.txt
├── models/              # Saved model checkpoints (created after training)
└── logs/                # Training logs for TensorBoard / plotting
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train a new agent:**
```bash
python train.py
```
Trains for 500,000 timesteps by default (~30-90 minutes on CPU, depending
on grid size). Saves the final model and the best-performing checkpoint
into `models/`.

**Watch a trained agent play:**
```bash
python play.py
# or specify a model explicitly:
python play.py models/dqn_snake_final.zip
```

**Plot training progress:**
```bash
python plot_progress.py
```

## State, Action, and Reward Design

### State (14 values, all 0 or 1)
The agent does not see the raw grid. It sees a compact feature vector,
computed relative to the snake's current head and direction:

- **Danger detection (6 values):** is there a wall or the snake's own
  body immediately ahead / to the right / to the left, and 2 cells
  further out in each of those directions
- **Current direction (4 values):** one-hot encoding of left/right/up/down
- **Food direction (4 values):** is the food to the left / right / above /
  below the head

### Actions (3 values)
Relative to the snake's current heading: `0 = straight`, `1 = turn right`,
`2 = turn left`.

### Reward
| Event | Reward |
|---|---|
| Eats food | +10 |
| Dies (wall or self-collision) | -10 |
| Each step (efficiency incentive) | -0.01 |
| Moves closer to food | +0.1 |
| Moves farther from food | -0.1 |
| Lands in a tight space (fewer than 10 reachable cells via flood-fill) | scaled penalty, up to -0.5 |

The last two rows are **reward shaping** — extra signals added on top of
the core win/lose reward to make learning faster and more stable:
- *Distance shaping* gives feedback every step instead of only at food/death,
  so the agent isn't relying on random wandering to discover that "toward
  food" is good.
- *Space-awareness shaping* (flood-fill) discourages the agent from
  boxing itself into a dead end with its own body — a common failure mode
  once the snake grows long. It's based on the number of empty cells
  reachable from the snake's new head position, compared to a fixed
  minimum safe threshold (not the snake's own length — see note below).

**Note on shaping choices:** shaping rewards were deliberately kept small
relative to the core +10/-10 signal. An earlier version compared reachable
space to the snake's own body length instead of a fixed threshold, which
misfired constantly on longer snakes (since total open space naturally
shrinks as the snake grows, regardless of actual danger) and measurably
hurt training. This is documented here because it's a realistic pitfall
in reward shaping generally, not just this project.

## Training Results

The agent was trained for 500,000 timesteps. Reward climbs sharply
between roughly 80k-110k timesteps as the network starts reliably
connecting state to useful action, then plateaus around an average
episode reward of ~200-230, with peak episodes reaching 450-500. Run
`plot_progress.py` after training to reproduce this curve.

## Known Limitations

- The state only looks 1-2 cells ahead; it has no long-range planning, so
  very long snakes can still occasionally trap themselves in situations
  the flood-fill penalty doesn't fully prevent.
- Performance depends on grid size (`config.py`) — a much larger grid
  gives the agent more room to make mistakes before running out of moves,
  and may need longer training to reach the same relative performance.

## Possible Future Improvements

- Double/Dueling DQN architectures for more accurate Q-value estimates
- Full-grid (CNN-based) state representation instead of hand-engineered features
- Curriculum learning: start on a small grid, increase size as performance improves
- PPO comparison (both algorithms are already switchable via `ALGORITHM` in `train.py`)