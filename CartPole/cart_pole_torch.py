
import numpy as np
import torch
import gymnasium as gym

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layers sizes")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs to train for")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("--evaluate_each", default=50, type=int, help="After how many epochs to evaluate")
parser.add_argument("--evaluate_for", default=10, type=int, help="For How many Epochs to evaluate")


class Agent(torch.nn.Module):
    
    def __init__(self, env: gym.Env, args: argparse.Namespace , *kwargs) -> None:
        super().__init__(*kwargs)
        
        self.inputs = torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer)
        self.observation_space = env.observation_space.shape[0]
        self.activation = torch.nn.ReLU()
        self.outputs = torch.nn.Linear(args.hidden_layer, env.action_space.n)
        self.action_space = env.action_space.n
        self.softmax = torch.nn.Softmax()
        
    def forward_loss(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor):
        res = self.inputs(states)
        res = self.activation(res)
        res = self.outputs(res)
        res = self.softmax(res)
        
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(res, actions)
        loss = loss * returns
        loss = loss.sum()
        
        
        return {"actions" : res, "loss" : loss}
    
    def forward(self, states: torch.Tensor):
        res = self.inputs(states)
        res = self.activation(res)
        res = self.outputs(res)
        res = self.softmax(res)
        return res
    
class Trainer:
    
    def __init__(self, model: Agent, env: gym.Env, epochs: int, batch_size:int, optimizer: torch.optim.Optimizer, evaluate_each:int = None, evaluate_for:int = 10) -> None:
        self.model = model
        self.env = env
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer= optimizer
        self.evaluate_each = evaluate_each
        self.evaluate_for = evaluate_for
        
    def evaluate(self, j):
        self.model.eval()
        rewards_count = []
        for _ in range(self.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                agent_prob = self.model.forward(torch.tensor(state))
                action = np.random.choice([0,1], p=torch.clone(agent_prob).detach().numpy())
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}") 
        
    def train(self):
        j = 0
        for _ in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(self.batch_size):
                states, actions, rewards = [], [], []
                state, done = self.env.reset()[0], False
                while not done:
                    agent_prob = self.model.forward(torch.tensor(state))
                    action = np.random.choice([0,1], p=torch.clone(agent_prob).detach().numpy())
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state
                    
                for i in reversed(range(len(rewards) - 1)):
                    rewards[i] +=  rewards[i + 1]
                    
            batch_states += states
            batch_actions += actions
            batch_returns += rewards
            
            returns = self.model.forward_loss(torch.tensor(batch_states), torch.tensor(batch_actions).type(torch.LongTensor), torch.tensor(batch_returns))
            returns["loss"].backward()
            self.optimizer.step()         
                      
            if self.evaluate_each != None and j % self.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = Agent(env, args)
    
    optimizer= torch.optim.AdamW(model.parameters(),lr=args.lr)
    trainer = Trainer(model, env, args.epochs, args.batch_size, optimizer, args.evaluate_each, args.evaluate_for)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make("CartPole-v1")
    
    main(env, args)
        