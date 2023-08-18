import numpy as np
import gymnasium as gym

class DescreteEnv:
    
    def __init__(self, env: gym.Env, tiles: int = 16) -> None:
        self.env = env
        self.tiles = tiles
        self.observation_space =tiles ** env.observation_space.shape[0]
        self.base_mesh = np.linspace(np.min(env.observation_space.low), np.max(env.observation_space.high), tiles)
        self.action_space = env.action_space
        
    def step(self, action):
        state_raw, reward, terminated, truncated, temp = self.env.step(action)
        counter = 0
        state = np.zeros(self.observation_space)
        for i in range(len(state_raw)):
            ids = np.argmin(np.abs(self.base_mesh - state_raw[i]))
            counter += ids * (self.tiles ** i)
        state[counter] = 1
        return state, reward, terminated, truncated, temp
    
    def reset(self):
        state_raw = self.env.reset()[0]
        counter = 0
        state = np.zeros(self.observation_space)
        for i in range(len(state_raw)):
            ids = np.argmin(np.abs(self.base_mesh - state_raw[i]))
            counter += ids * (self.tiles ** i)
        state[counter] = 1
        return state.reshape(1,-1)
        
        
        
        