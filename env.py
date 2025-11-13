import numpy as np
import gym
from gym import spaces

class ContinuousMazeEnv(gym.Env):
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0.0,0.0],dtype=np.float32),
            high=np.array([1.0,1.0],dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.state = None
        
        self.goal = np.array([0.9,0.5],dtype=np.float32)
        self.goal_radius = 0.05
        self.danger_zones = [(0.4,0.85,0.6,0.9), (0.4,0.1,0.6,0.15), (0.45,0.48,0.55,0.52)]
        self.walls = [(0.3,0.9,0.7,1.0), (0.3,0.0,0.7,0.1)]
        self.step_size = 0.05

    def reset(self, seed=None, options=None):
        self.state = np.array([0.1,0.5],dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x,y = self.state
        if action==0: y+=self.step_size
        elif action==1: y-=self.step_size
        elif action==2: x-=self.step_size
        elif action==3: x+=self.step_size
        x = np.clip(x,0,1);  y = np.clip(y,0,1)
        new_state = np.array([x,y],dtype=np.float32)

        
        reward = -0.01  
        done = False
        
        
        for (xmin,ymin,xmax,ymax) in self.walls:
            if xmin<=x<=xmax and ymin<=y<=ymax:
                reward -= 2.0
                new_state = self.state.copy()
                break
        
        
        for (xmin,ymin,xmax,ymax) in self.danger_zones:
            if xmin<=new_state[0]<=xmax and ymin<=new_state[1]<=ymax:
                reward -= 10.0
                done = True
                break


        if np.linalg.norm(new_state-self.goal)<=self.goal_radius:
            reward += 10.0
            done = True

        self.state = new_state
        return self.state.copy(), reward, done, False, {}


    def render(self, mode="human"):
        import pygame
        if not hasattr(self, "screen"):
            pygame.init()
            self.size = 500
            self.screen = pygame.display.set_mode((self.size, self.size))
            pygame.display.set_caption("Continuous Maze Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        
        for (xmin, ymin, xmax, ymax) in self.walls:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                pygame.Rect(
                    int(xmin*self.size),
                    self.size - int(ymax*self.size),
                    int((xmax - xmin)*self.size),
                    int((ymax - ymin)*self.size)
                )
            )

        
        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            pygame.draw.rect(
                self.screen, (255, 0, 0),
                pygame.Rect(
                    int(xmin*self.size),
                    self.size - int(ymax*self.size),
                    int((xmax - xmin)*self.size),
                    int((ymax - ymin)*self.size)
                )
            )

        
        pygame.draw.circle(
            self.screen, (0, 255, 0),
            (int(self.goal[0]*self.size), self.size - int(self.goal[1]*self.size)),
            int(self.goal_radius*self.size)
        )

        
        pygame.draw.circle(
            self.screen, (0, 0, 255),
            (int(self.state[0]*self.size), self.size - int(self.state[1]*self.size)),
            10
        )

        pygame.display.flip()
        self.clock.tick(30)  

    def close(self):
        import pygame
        if hasattr(self, "screen"):
            pygame.quit()
            del self.screen

