
import random
import tensorflow as tf
import argparse
import gymnasium as gym
import numpy as np
import collections

class TD3:
    
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
        self.actor = TD3.Actor(args, env)
        self.actor.compile(optimizer= "adam")
        
        self.target_actor = TD3.Actor(args, env)
        self.target_actor.compile()
        
        self.critic = TD3.Critic(args, env)
        self.critic.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        self.target_critic = TD3.Critic(args, env)
        self.target_critic.compile()
        
        self.other_critic = TD3.Critic(args, env)
        self.other_critic.compile(
            optimizer="adam",
            loss = tf.keras.losses.MeanSquaredError()
        )
        
        self.other_target_critic = TD3.Critic(args, env)
        self.other_target_critic.compile()
        
        self.target_forgetting = args.target_forgetting


        
    @tf.function(experimental_relax_shapes=True)    
    def train(self, states, actions, returns):
        with tf.GradientTape() as actor_tape:
            actor_actions =  self.actor(states)
            
            critic_val = self.critic(tf.concat([states, actor_actions], 1))
            
            actor_loss = -tf.reduce_mean(critic_val)
        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        
        
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
        
        with tf.GradientTape() as critic_tape:          
            predict_critic = self.critic(tf.concat([states, actions], 1))
            critic_loss = self.critic.loss(returns, predict_critic)
            
        critic_grad = critic_tape.gradient(critic_loss, self.critic.trainable_variables)       
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        with tf.GradientTape() as other_critic_tape:
            predict_other_critic = self.other_critic(tf.concat([states, actions], 1))
            other_critic_loss = self.other_critic.loss(returns, predict_other_critic)
            
        other_critic_grad = other_critic_tape.gradient(other_critic_loss, self.other_critic.trainable_variables)       
        self.other_critic.optimizer.apply_gradients(zip(other_critic_grad, self.other_critic.trainable_variables))
        
        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
            
        for var, target_var in zip(self.other_critic.trainable_variables, self.other_target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
            
    @tf.function()
    def actor_forward(self, states):
        return self.actor(states)
    
    @tf.function()
    def critic_forward(self, states):
        actor_actions = tf.clip_by_value(self.target_actor(states) + np.random.normal(scale =0.1), self.actor.env.action_space.low, self.actor.env.action_space.high)
        return tf.minimum(self.target_critic(tf.concat([states, actor_actions], 1)), self.other_target_critic(tf.concat([states, actor_actions], 1)))
         
    
class Trainer:
    
    def __init__(self, agent: TD3,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = self.agent.actor_forward(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]
                    
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
            
            state, done = self.env.reset()[0], False
            while not done:
                action = np.clip(self.agent.actor_forward
                                 (np.asarray(state, dtype=np.float32).reshape(1, -1))[0] + np.random.normal(scale =0.1), 
                                 self.env.action_space.low, 
                                 self.env.action_space.high)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                replay_buffer.append((state, action, reward, done, next_state))

                state = next_state
                
                if len(replay_buffer) >= self.args.min_buffer:
                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[snapshot for snapshot in episode]))
                    returns = rewards[:,None] + self.args.gamma * self.agent.critic_forward(np.asarray(next_states, dtype=np.float32).reshape(self.args.batch_size, -1)) * np.logical_not(dones)[:,None]
                    self.agent.train(states.astype(np.float32), actions.astype(np.float32), returns)  

            
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = TD3(args, env)
    
    trainer = Trainer(model, env, args)
    trainer.train()
