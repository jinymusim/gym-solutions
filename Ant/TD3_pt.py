
import random
import torch
import argparse
import gymnasium as gym
import numpy as np
import collections

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="Ant-v4", type=str, help="Environment.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_forgetting", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--min_buffer", default=256, type=int, help="Training episodes")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate.")




class TD3:
    
    class Actor(torch.nn.Module):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.observation_space = env.observation_space.shape[0]
            self.action_space = env.action_space.shape[0]
            
            self.hidden = torch.nn.Linear(self.observation_space, args.hidden_size)
            self.relu = torch.nn.ReLU()
            self.mean = torch.nn.Linear(args.hidden_size, self.action_space)
            self.env = env
            
        def forward(self, inputs: torch.Tensor):
            hidden = self.hidden(inputs)
            hidden = self.relu(hidden)
            means = self.mean(hidden)   
            means = torch.clip(means,torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high))
                     
            return means
        
    class Critic(torch.nn.Module):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.observation_space = env.observation_space.shape[0]
            self.action_space = env.action_space.shape[0]
            
            self.hidden = torch.nn.Linear(self.observation_space + self.action_space, args.hidden_size)
            self.relu = torch.nn.ReLU()
            self.output_val = torch.nn.Linear(args.hidden_size, 1)
            
        def forward(self, inputs: torch.Tensor):
            hidden = self.hidden(inputs)
            hidden = self.relu(hidden)
            outputs = self.output_val(hidden)
            return outputs
        
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        self.actor = TD3.Actor(args, env)
        
        self.target_actor = TD3.Actor(args, env)
        
        self.critic = TD3.Critic(args, env)   
        self.target_critic = TD3.Critic(args, env)
        
        self.other_critic = TD3.Critic(args, env)  
        self.other_target_critic = TD3.Critic(args, env)
        
        self.target_forgetting = args.target_forgetting

   
    def loss(self, states, actions, returns):
          
        actor_actions =  self.actor.forward(states)           
        critic_val = self.critic.forward(torch.concat([states, actor_actions], 1))
        actor_loss = -torch.mean(critic_val)
                
        predict_critic = self.critic.forward(torch.concat([states, actions], 1))
        critic_loss_fnc = torch.nn.MSELoss()
        critic_loss = critic_loss_fnc(predict_critic, returns)
        
        other_critic_predict = self.other_critic.forward(torch.concat([states, actions], 1))
        other_critic_loss_fnc = torch.nn.MSELoss()
        other_critic_loss = other_critic_loss_fnc(other_critic_predict, returns)
        
        return  {"actor_loss" : actor_loss,
                 "critic_loss" : critic_loss,
                 "other_critic_loss" : other_critic_loss
            }
        
        
    def close_up_target(self):
        
        for var, target_var in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_var.data.copy_(target_var.data * (1 - self.target_forgetting) + var.data * self.target_forgetting)
        
        for var, target_var in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_var.data.copy_(target_var.data * (1 - self.target_forgetting) + var.data * self.target_forgetting)
            
        for var, target_var in zip(self.other_critic.parameters(), self.other_target_critic.parameters()):
            target_var.data.copy_(target_var.data * (1 - self.target_forgetting) + var.data * self.target_forgetting)
            
    def eval(self):
        self.actor.train(False)
        self.target_actor.train(False)
        self.critic.train(False)
        self.target_critic.train(False)
        self.other_critic.train(False)
        self.other_target_critic.train(False)
    
    def train(self):
        self.actor.train(True)
        self.target_actor.train(False)
        self.critic.train(True)
        self.target_critic.train(False)
        self.other_critic.train(True)
        self.other_target_critic.train(False)
        
            
    def actor_forward(self, states):
        return self.actor.forward(states.type(torch.FloatTensor))
    
    def critic_forward(self, states):
        actor_actions = torch.clip(self.target_actor.forward(states.type(torch.FloatTensor)) + torch.normal(mean=0.0,std =0.1, size=(states.shape[0], self.actor.action_space)),
                                   torch.tensor(self.actor.env.action_space.low), 
                                   torch.tensor(self.actor.env.action_space.high)
                                )
        return torch.minimum(
            self.target_critic.forward(
                torch.concat([states.type(torch.FloatTensor), actor_actions.type(torch.FloatTensor)], 1)
            ), 
            self.other_target_critic(
                torch.concat([states.type(torch.FloatTensor), actor_actions.type(torch.FloatTensor)], 1)
            )
        )
                             
         
    
class Trainer:
    
    def __init__(self, agent: TD3,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        self.actor_optimizer = torch.optim.AdamW(self.agent.actor.parameters(),lr=args.lr)
        self.critic_optimizer = torch.optim.AdamW(self.agent.critic.parameters(),lr=args.lr)
        self.other_critic_optimizer = torch.optim.AdamW(self.agent.other_critic.parameters(),lr=args.lr)
        
    def evaluate(self, j):
        self.agent.eval()
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = self.agent.actor_forward(torch.tensor(state, dtype=torch.float32).view(1, -1))[0]
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        j = 0
        
        
        replay_buffer = collections.deque(maxlen=self.args.replay_buffer_size)
        self.agent.eval()
        while True:
            
            state, done = env.reset()[0], False
            while not done:
                action = torch.clip(
                    self.agent.actor_forward(torch.tensor(state).view(1, -1)) + torch.normal(mean=0.0,std =0.1, size=(1,self.env.action_space.shape[0])), 
                    torch.tensor(self.env.action_space.low), 
                    torch.tensor(self.env.action_space.high)
                    )
                
                next_state, reward, terminated, truncated, _ = self.env.step(action.detach().numpy()[0])
                done = terminated or truncated

                replay_buffer.append((state, action.detach().numpy(), reward, done, next_state))

                state = next_state
                
                if len(replay_buffer) >= self.args.min_buffer:
                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    states, actions, rewards, dones, next_states = map(torch.tensor, zip(*[snapshot for snapshot in episode]))
                    returns = rewards[:,None] + self.args.gamma * self.agent.critic_forward(next_states.view(self.args.batch_size, -1)) * torch.logical_not(dones)[:,None]
                    
                    self.agent.train()
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    self.other_critic_optimizer.zero_grad()
                    losses =  self.agent.loss(states.type(torch.FloatTensor),
                                              actions.type(torch.FloatTensor).view(self.args.batch_size, self.env.action_space.shape[0]), 
                                              returns.type(torch.FloatTensor))  
                    losses["actor_loss"].backward(retain_graph=True)
                    losses["critic_loss"].backward(retain_graph=True)
                    losses["other_critic_loss"].backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    self.other_critic_optimizer.step()
                    self.agent.close_up_target()
                    self.agent.eval()
            
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = TD3(args, env)

    trainer = Trainer(model, env, args)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make(args.env)
    
    main(env, args)
