import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os  

class ForestForageEnv(gym.Env):
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([0, 0])
        self.berry_positions = [(1, 1), (3, 3), (4, 0), (1, 4)]
        self.trap_positions = [(1, 3), (3, 2)]
        self.collected = set()
        self.max_steps = 500
        self.step_count = 0

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

        
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        
        self.berry_img = mpimg.imread(os.path.join(self.current_dir, "Berries.png"))
        self.trap_img = mpimg.imread(os.path.join(self.current_dir, "bear-trap-.png"))

    def reset(self, seed=None, options=None):
        self.agent_state = np.array([0, 0])
        self.collected = set()
        self.step_count = 0
        return self.agent_state, {}

    def step(self, action):
        x, y = self.agent_state

        if action == 0 and y < self.grid_size - 1:  # up
            y += 1
        elif action == 1 and y > 0:  # down
            y -= 1
        elif action == 2 and x > 0:  # left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # right
            x += 1

        self.agent_state = np.array([x, y])
        self.step_count += 1

        reward = 0
        done = False

        if tuple(self.agent_state) in self.trap_positions:
            reward = -10
            done = True
        elif tuple(self.agent_state) in self.berry_positions and tuple(self.agent_state) not in self.collected:
            reward = 10
            self.collected.add(tuple(self.agent_state))
            if len(self.collected) == len(self.berry_positions):
                done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {"berries_collected": len(self.collected)}
        return self.agent_state, reward, done, False, info

    def render(self):
        self.ax.clear()
        self.ax.set_xticks(np.arange(0, self.grid_size + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(0, self.grid_size + 1) - 0.5, minor=True)
        self.ax.grid(which="minor", color='gray', linestyle='-', linewidth=1)
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")

        
        self.ax.plot(self.agent_state[0], self.agent_state[1], "ro", label="Agent")

        
        berry_icon = OffsetImage(self.berry_img, zoom=0.08)
        for berry in self.berry_positions:
            if berry not in self.collected:
                berry_box = AnnotationBbox(berry_icon, berry, frameon=False)
                self.ax.add_artist(berry_box)

        
        trap_icon = OffsetImage(self.trap_img, zoom=0.15)
        for trap in self.trap_positions:
            trap_box = AnnotationBbox(trap_icon, trap, frameon=False)
            self.ax.add_artist(trap_box)

        plt.pause(0.2)

    def close(self):
        plt.close()

if __name__ == "__main__":
    env = ForestForageEnv()
    state, _ = env.reset()

    for i in range(500):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        env.render()
        print(f"Step: {i}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("🍓 Done! Either trapped or all berries collected!")
            break

    env.close()
