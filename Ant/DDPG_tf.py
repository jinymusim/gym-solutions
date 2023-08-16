
import random
import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import gymnasium as gym
import numpy as np
import collections

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="Ant-v4", type=str, help="Environment.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_size", default=16, type=int, help="Size of hidden layer.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_forgetting", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--min_buffer", default=64, type=int, help="Training episodes")




class DDPG:
    
    class Actor(tf.keras.Model):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.mean = tf.keras.layers.Dense(env.action_space.shape[0], activation=None)
            self.env = env
            
        def call(self, inputs: tf.Tensor):
            hidden = self.hidden(inputs)
            means = self.mean(hidden)   
            means = tf.clip_by_value(means, self.env.action_space.low, self.env.action_space.high)
            
            
            return means
        
    class Critic(tf.keras.Model):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.output_val = tf.keras.layers.Dense(1, activation=None)
            
        def call(self, inputs:tf.Tensor):
            hidden = self.hidden(inputs)
            outputs = self.output_val(hidden)
            return outputs
        
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        self.actor = DDPG.Actor(args, env)
        self.actor.compile(optimizer= "adam")
        
        self.critic = DDPG.Critic(args, env)
        self.critic.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        self.target_critic = DDPG.Critic(args, env)
        self.target_critic.compile()
        
        self.target_forgetting = args.target_forgetting


        
    @tf.function(experimental_relax_shapes=True)    
    def train(self, states, actions, returns):
        with tf.GradientTape() as actor_tape:
            actor_actions =  self.actor(states)
            
            critic_val = self.critic(tf.concat([states, actor_actions], 1))
            
            actor_loss = -tf.reduce_mean(critic_val)
        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        
        with tf.GradientTape() as critic_tape:          
            # Predicted values
            predict_critic = self.critic(tf.concat([states, actions], 1))
            # Loss already returns mean 
            critic_loss = self.critic.loss(returns, predict_critic)
        critic_grad = critic_tape.gradient(critic_loss, self.critic.trainable_variables)       
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
            
    @tf.function()
    def actor_forward(self, states):
        return self.actor(states)
    
    @tf.function()
    def critic_forward(self, states):
        actor_actions = self.actor(states)
        return self.critic(tf.concat([states, actor_actions], 1))
         
    
class Trainer:
    
    def __init__(self, agent: DDPG,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = self.agent.actor_forward(np.asarray(state,dtype=np.float32).reshape(1, -1))[0]
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        j = 0
        
        
        replay_buffer = collections.deque(maxlen=self.args.replay_buffer_size)
        
        while True:
            
            state, done = env.reset()[0], False
            while not done:
                action = np.clip(self.agent.actor_forward(np.asarray(state, dtype=np.float32).reshape(1, -1))[0] * np.random.normal(scale =0.1), self.env.action_space.low, self.env.action_space.high)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                replay_buffer.append((state, action, reward, done, next_state))

                state = next_state
                
                if len(replay_buffer) >= self.args.min_buffer:
                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[snapshot for snapshot in episode]))
                    returns = rewards[:,None] + self.args.gamma * self.agent.critic_forward(np.asarray(next_states, dtype=np.float32).reshape(self.args.batch_size, -1))[0] * np.logical_not(dones)[:,None]
                    self.agent.train(states.astype(np.float32), actions.astype(np.float32), returns)  

            
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = DDPG(args, env)
    
    trainer = Trainer(model, env, args)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make(args.env)
    
    main(env, args)
