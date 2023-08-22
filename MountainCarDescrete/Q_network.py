
import numpy as np
import gymnasium as gym
import collections
import random
from discretise import DiscreteEnv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MountainCar-v0", type=str, help="Environment.")
parser.add_argument("--hidden_layer", default=32, type=int, help="Hidden layers sizes")
parser.add_argument("--epochs", default=5000, type=int, help="Number of epochs to train for")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("--evaluate_each", default=50, type=int, help="After how many epochs to evaluate")
parser.add_argument("--evaluate_for", default=10, type=int, help="For How many Epochs to evaluate")
parser.add_argument("--epsilon", default=0.9, type=float, help="Exploration factor")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--min_buffer", default=100, type=int, help="Training episodes")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiling per dimension")

class Q_Network:
    
    def __init__(self, env: DiscreteEnv, args: argparse.Namespace) -> None:
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space.n
        self.network = np.zeros((self.observation_space, self.action_space))
        

class Trainer:
    
    def __init__(self, agent: Q_Network, env: DiscreteEnv, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = np.argmax(self.agent.network[state, :])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        
        
        epsilon = self.args.epsilon
        j = 0
        for epoch in range(self.args.epochs):
            state, done = self.env.reset()[0], False
            while not done:
                
                if random.random() < epsilon:
                    action = random.choice(list(range(self.env.action_space.n)))
                else:
                    action = np.argmax(self.agent.network[state, :])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.agent.network[state, action] += self.args.lr * (reward + (1 - done) * self.args.gamma * (np.max(self.agent.network[next_state, :]) - self.agent.network[state, action]) )

                state = next_state
                
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            if self.args.epsilon_final_at:
                epsilon = np.interp(epoch + 1, [0, self.args.epsilon_final_at], [self.args.epsilon, self.args.epsilon_final])

            
def main(env, args):
    model = Q_Network(env, args)
    
    trainer = Trainer(model, env, args)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make(args.env)
    env = DiscreteEnv(env, args.tiles)
    
    main(env, args)
        