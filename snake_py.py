import numpy as np
import pygame
import random
from collections import deque
from enum import Enum

import gymnasium as gym
from gymnasium import spaces


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

#for makeing the ' train grid ' and '  play grid ' in equal size
class Gridsize():

    def assign_grid_size(self,grid_w=30, grid_h=30):
        GRID_W = grid_w
        GRID_H = grid_h
        return (GRID_W, GRID_H)




# (dx, dy) for each direction, in grid cells
DIRECTION_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}

BLOCK_SIZE = 20


class SnakeEnv(gym.Env):
    """
    used
    Gymnasium environment for  Snake start.

    Actions (relative to current heading):
        0 = go straight
        1 = turn right
        2 = turn left
    """

    metadata = {"render_modes": ["human", None], "render_fps": 40}


    def __init__(self, grid_w=20, grid_h=20, render_mode=None):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)

        self.window = None
        self.clock = None
        self.font = None

        self.max_steps_without_food = grid_w * grid_h * 4  # prevents infinite loitering

        self.reset()


    # Core Gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.direction = Direction.RIGHT
        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.head = self.snake[0]

        self.score = 0
        self.steps_since_food = 0
        self.frame_count = 0

        self._place_food()

        obs = self._get_state()
        info = {"score": self.score}
        return obs, info

    def step(self, action):
        self.frame_count += 1
        self.steps_since_food += 1

        prev_head = self.snake[0]
        old_distance = self._manhattan(prev_head, self.food)

        self._move(action)  # updates self.head, pushes into snake

        reward = -0.01  # small per-step penalty encourages efficient paths
        terminated = False
        truncated = False

        if self._is_collision():
            terminated = True
            reward = -10
            self.snake.popleft() if False else None  # no-op, head already invalid
            obs = self._get_state()
            info = {"score": self.score}
            return obs, reward, terminated, truncated, info

        # --- Distance-based reward shaping ---
        # Small nudge toward food on every step, not just when it's eaten.
        # Kept deliberately small (+/-0.1) relative to the +10/-10 food/death
        # rewards so it guides behavior without overriding the real signal
        # (unlike the earlier flood-fill penalty bug).
        new_distance = self._manhattan(self.head, self.food)
        if new_distance < old_distance:
            reward += 0.1
        elif new_distance > old_distance:
            reward -= 0.1

        self.snake.appendleft(self.head)

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()  # move forward: drop tail

        # --- Space-awareness penalty ---
        # Flood-fill the open space reachable from the new head. We compare
        # against a small FIXED threshold, not the snake's own length --
        # comparing to body length fires constantly once the snake covers a
        # large fraction of the board (common on bigger grids), punishing
        
        # tight corner, regardless of how long the snake has grown.
        SAFE_SPACE_THRESHOLD = 10
        reachable = self._flood_fill(self.head)
        if reachable < SAFE_SPACE_THRESHOLD:
            # scale the penalty by how tight the spot is (0 room = full penalty)
            reward -= 0.5 * (1 - reachable / SAFE_SPACE_THRESHOLD)

        if self.steps_since_food > self.max_steps_without_food:
            truncated = True

        obs = self._get_state()
        info = {"score": self.score}
        return obs, reward, terminated, truncated, info


    # Game mechanics

    def _move(self, action):
        # action: 0 = straight, 1 = right turn, 2 = left turn
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == 1:  # turn right
            idx = (idx + 1) % 4
        elif action == 2:  # turn left
            idx = (idx - 1) % 4
        # action == 0: keep idx (straight)

        self.direction = clockwise[idx]
        dx, dy = DIRECTION_VECTORS[self.direction]
        hx, hy = self.snake[0]
        self.head = (hx + dx, hy + dy)

    def _is_collision(self, point=None):
        point = self.head if point is None else point
        x, y = point
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True
        if point in list(self.snake)[:-1]:  # tail cell will move away, ignore it
            return True
        return False

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _flood_fill(self, start):
        """
        Counts how many empty cells are reachable from `start` via
        up/down/left/right moves, treating the snake's body as walls.
        This tells us how much "room to maneuver" is left after a move --
        a small number means the snake may be boxing itself in.

        Note: `start` is normally the snake's current head, which is
        already part of self.snake -- we must exclude it from the
        blocked set, or every call would immediately see its own
        starting cell as "blocked by the snake" and return 0.
        """
        blocked = set(self.snake)
        blocked.discard(start)
        visited = {start}
        stack = [start]
        count = 0

        while stack:
            x, y = stack.pop()
            count += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_w and 0 <= ny < self.grid_h
                        and (nx, ny) not in blocked
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    stack.append((nx, ny))

        return count

    def _place_food(self):
        while True:
            fx = random.randint(0, self.grid_w - 1)
            fy = random.randint(0, self.grid_h - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                return


    # State encoding

    def _get_state(self):
        hx, hy = self.head
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        dir_straight = self.direction
        dir_right = clockwise[(idx + 1) % 4]
        dir_left = clockwise[(idx - 1) % 4]

        def point_in_dir(d):
            dx, dy = DIRECTION_VECTORS[d]
            return (hx + dx, hy + dy)

        danger_straight = self._is_collision(point_in_dir(dir_straight))
        danger_right = self._is_collision(point_in_dir(dir_right))
        danger_left = self._is_collision(point_in_dir(dir_left))

        # 2-step lookahead: danger two cells further out in each direction.
        #checking the snake danger a head of 2 steps
        # gives useful early warning of nearby body/walls.
        def point_two_ahead(d):
            dx, dy = DIRECTION_VECTORS[d]
            return (hx + dx * 2, hy + dy * 2)

        danger2_straight = self._is_collision(point_two_ahead(dir_straight))
        danger2_right = self._is_collision(point_two_ahead(dir_right))
        danger2_left = self._is_collision(point_two_ahead(dir_left))

        fx, fy = self.food

        state = [
            danger_straight,
            danger_right,
            danger_left,
            danger2_straight,
            danger2_right,
            danger2_left,
            self.direction == Direction.LEFT,
            self.direction == Direction.RIGHT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,
            fx < hx,  # food left
            fx > hx,  # food right
            fy < hy,  # food up
            fy > hy,  # food down
        ]
        return np.array(state, dtype=np.float32)

    '''
    this contains the prasent state values or observations :
     Observation (11 floats, all 0/1):
        [danger_straight, danger_right, danger_left,
         dir_left, dir_right, dir_up, dir_down,
         food_left, food_right, food_up, food_down]
    '''


    # Rendering

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_w * BLOCK_SIZE, self.grid_h * BLOCK_SIZE)
            )
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return

        self.window.fill((20, 20, 20))

        for i, (x, y) in enumerate(self.snake):
            color = (0, 200, 0) if i == 0 else (0, 140, 0)
            pygame.draw.rect(
                self.window, color,
                (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            )

        fx, fy = self.food
        pygame.draw.rect(
            self.window, (200, 30, 30),
            (fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
        )

        score_surf = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_surf, (5, 5))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None