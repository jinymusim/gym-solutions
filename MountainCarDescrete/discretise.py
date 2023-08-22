import numpy as np
import gymnasium as gym

class DiscreteEnv:
    
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
        return state[np.newaxis, :]
    
class DiscreteVenv:
    
    def __init__(self,venv : gym.vector.VectorEnv, tiles: int = 16) -> None:
        self.venv = venv
        self.tiles = tiles
        self.observation_space =tiles ** venv.single_observation_space.shape[0]
        self.base_mesh = np.linspace(np.min(venv.single_observation_space.low), np.max(venv.single_observation_space.high), tiles)
        self.action_space = venv.single_action_space
        
    
    
    def reset(self):
        states_raw = self.venv.reset()[0]
        states = np.zeros((states_raw.shape[0], self.observation_space))
        for j in range(states.shape[0]):
            counter = 0
            for i in range(len(states_raw[j])):
                ids = np.argmin(np.abs(self.base_mesh - states_raw[j,i]))
                counter += ids * (self.tiles ** i)
            states[j,counter] = 1
        return states[np.newaxis, :, :]
    
    def step(self, action):
        states_raw, reward, terminated, truncated, temp = self.venv.step(action)
        states = np.zeros((states_raw.shape[0], self.observation_space))
        for j in range(states.shape[0]):
            counter = 0
            for i in range(len(states_raw[j])):
                ids = np.argmin(np.abs(self.base_mesh - states_raw[j,i]))
                counter += ids * (self.tiles ** i)
            states[j,counter] = 1
        return states, reward, terminated, truncated, temp
    
        
        
        
        