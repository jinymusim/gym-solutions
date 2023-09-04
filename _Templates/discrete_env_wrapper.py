import numpy as np
import gymnasium as gym

class DiscreteEnv(gym.Wrapper):
    
    def __init__(self, env: gym.Env, tiles: int = 16) -> None:
        super().__init__(env)
        self.env = env
        self.tiles = tiles
        self.observation_types = env.observation_space.shape[0]
        self.observation_space =tiles * env.observation_space.shape[0]
        self.base_meshes = [np.linspace(env.observation_space.low[i], env.observation_space.high[i], tiles) for i in range(self.observation_types)] 
        
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