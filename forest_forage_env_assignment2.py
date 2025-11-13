import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

class ForestForageEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        grid_size: int = 5,
        berry_image: str = "Berries.png",
        trap_image: str = "bear-trap-.png",
        max_steps: int = 150
    ):
        super().__init__()

        # 1) Grid parameters & agent start
        self.grid_size = grid_size
        self.agent_state = np.array([0, 0], dtype=int)

        # 2) Berry & trap positions
        self.berry_positions = [(1, 1), (3, 3), (4, 0), (1, 4)]
        self.trap_positions  = [(1, 3), (3, 2)]

        # 3) Bookkeeping
        self.collected = set()
        self.max_steps = max_steps
        self.step_count = 0

        # 4) Define action & observation spaces
        self.action_space = gym.spaces.Discrete(4)  # 0=up,1=down,2=left,3=right
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

        # 5) Matplotlib figure setup (for render)
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

        # 6) Load berry & trap images (must exist in the same folder)
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        berry_path = os.path.join(self.current_dir, berry_image)
        trap_path  = os.path.join(self.current_dir, trap_image)

        if not os.path.isfile(berry_path):
            raise FileNotFoundError(f"Could not find berry image at {berry_path}")
        if not os.path.isfile(trap_path):
            raise FileNotFoundError(f"Could not find trap image at {trap_path}")

        # ← These two lines load your berry and trap images:
        self.berry_img = mpimg.imread(os.path.join(self.current_dir, "Berries.png"))
        self.trap_img  = mpimg.imread(os.path.join(self.current_dir, "bear-trap-.png"))


    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state:
          - Agent back to (0,0)
          - No berries collected
          - step_count reset to 0
        Returns: (observation, info)
        """
        self.agent_state = np.array([0, 0], dtype=int)
        self.collected   = set()
        self.step_count  = 0
        return self.agent_state, {}

    def step(self, action):
        """
        Take one action (0=up,1=down,2=left,3=right).
        Returns: (observation, reward, done, truncated, info)
        """
        x, y = self.agent_state

        # 1) Move (with boundary checks)
        if action == 0 and y < self.grid_size - 1:    # up
            y += 1
        elif action == 1 and y > 0:                   # down
            y -= 1
        elif action == 2 and x > 0:                   # left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:   # right
            x += 1

        self.agent_state = np.array([x, y], dtype=int)
        self.step_count += 1

        # 2) Default reward: stronger step penalty to encourage speed
        reward = -0.1
        done = False

        # 3) Check if stepped on a trap: immediate failure
        if (x, y) in self.trap_positions:
            reward = -10
            done   = True

        # 4) Check if collected a new berry
        elif (x, y) in self.berry_positions and (x, y) not in self.collected:
            reward = 10
            self.collected.add((x, y))
            # If that was the last berry, success → end episode
            if len(self.collected) == len(self.berry_positions):
                done = True

        # 5) Check step limit: if time’s up before all berries, apply extra penalty
        if self.step_count >= self.max_steps:
            if len(self.collected) < len(self.berry_positions):
                reward = -5   # additional penalty for failing to collect all
            done = True

        info = {"berries_collected": len(self.collected)}
        return self.agent_state, reward, done, False, info

    def render(self):
        """
        Render the current grid, berries, traps, and agent. Uses Matplotlib:
          - Faint grid lines via minor ticks
          - Bear-trap icon at each trap position
          - Berry icon at each uncollected berry position
          - Agent as a red circle
        """
        self.ax.clear()

        
        self.ax.set_xticks(np.arange(0, self.grid_size + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(0, self.grid_size + 1) - 0.5, minor=True)
        self.ax.grid(which="minor", color='gray', linestyle='-', linewidth=1)
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")

        # 1) Draw agent as a red circle
        axp, ayp = self.agent_state
        self.ax.plot(axp, ayp, "ro", markersize=12)

        # 2) Draw any remaining berries (green icon)
        berry_icon = OffsetImage(self.berry_img, zoom=0.08)
        for (bx, by) in self.berry_positions:
            if (bx, by) not in self.collected:
                box = AnnotationBbox(berry_icon, (bx, by), frameon=False)
                self.ax.add_artist(box)

        # 3) Draw traps (bear-trap icon) at each trap position
        trap_icon = OffsetImage(self.trap_img, zoom=0.15)
        for (tx, ty) in self.trap_positions:
            box = AnnotationBbox(trap_icon, (tx, ty), frameon=False)
            self.ax.add_artist(box)

        plt.pause(1 / self.metadata["render_fps"])

    def close(self):
        """
        Close the rendering window.
        """
        plt.close()


# (Optional) Register as a Gym environment so you can call:
# gym.make("ForestForage-v0")
from gymnasium.envs.registration import register
register(
    id="ForestForage-v0",
    entry_point="forest_forage_env_assignment2:ForestForageEnv",
)
