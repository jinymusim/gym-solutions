import numpy as np
import gymnasium as gym

class DiscreteEnv:
    
    def __init__(self, env: gym.Env, tiles: int = 16) -> None:
        self.env = env
        self.tiles = tiles
        self.observation_types = env.observation_space.shape[0]
        self.observation_space =tiles * env.observation_space.shape[0]
        self.base_meshes = [np.linspace(env.observation_space.low[i], env.observation_space.high[i], tiles) for i in range(self.observation_types)] 
        self.action_space = env.action_space
        
    def step(self, action):
        state_raw, reward, terminated, truncated, temp = self.env.step(action)
        state = np.zeros((self.observation_types, self.tiles), dtype=np.int32)
        for i in range(self.observation_types):
            id = np.argmin(np.abs(self.base_meshes[i] - state_raw[i]))
            state[i,id] = 1
        return state.reshape(-1) , reward, terminated, truncated, temp
    
    def reset(self):
        state_raw = self.env.reset()[0]
        state = np.zeros((self.observation_types, self.tiles), dtype=np.int32)
        for i in range(self.observation_types):
            id = np.argmin(np.abs(self.base_meshes[i] - state_raw[i]))
            state[i,id] = 1
        return state.reshape(1,-1)
    
class DiscreteVenv:
    
    def __init__(self,venv : gym.vector.VectorEnv, tiles: int = 16) -> None:
        self.venv = venv
        self.tiles = tiles
        self.observation_types = venv.single_observation_space.shape[0]
        self.observation_space =tiles * venv.single_observation_space.shape[0]
        self.base_meshes = [np.linspace(venv.single_observation_space.low[i], venv.single_observation_space.high[i], tiles) for i in range(self.observation_types)] 
        self.action_space = venv.single_action_space
        
    
    
    def reset(self):
        states_raw = self.venv.reset()[0]
        states = np.zeros((states_raw.shape[0], self.observation_types, self.tiles), dtype=np.int32)
        for j in range(states.shape[0]):
            for i in range(self.observation_types):
                ids = np.argmin(np.abs(self.base_meshes[i] - states_raw[j,i]))
                states[j,i,ids] = 1
        return states.reshape(1, states_raw.shape[0], -1)
    
    def step(self, action):
        states_raw, reward, terminated, truncated, temp = self.venv.step(action)
        states = np.zeros((states_raw.shape[0], self.observation_types, self.tiles), dtype=np.int32)
        for j in range(states.shape[0]):
            for i in range(self.observation_types):
                ids = np.argmin(np.abs(self.base_meshes[i] - states_raw[j,i]))
                states[j,i,ids] = 1
        return states.reshape(states_raw.shape[0], -1), reward, terminated, truncated, temp