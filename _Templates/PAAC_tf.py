
import tensorflow as tf
import argparse
import gymnasium as gym
import numpy as np





class PAAC:
    
    class Actor(tf.keras.Model):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.actions  = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
            
        def call(self, inputs: tf.Tensor):
            hidden = self.hidden(inputs)
            actions = self.actions(hidden)   
                       
            return actions
        
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
        self.actor = PAAC.Actor(args, env)
        self.actor.compile(
            optimizer= "adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        )
        
        self.critic = PAAC.Critic(args, env)
        self.critic.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        self.entropy_regularization = args.entropy_regularization
        


        
    @tf.function(experimental_relax_shapes=True)    
    def train(self, states, actions, returns):
        
        
        with tf.GradientTape() as critic_tape:          
            predict_critic = self.critic(states)
            critic_loss = self.critic.loss(returns, predict_critic)
        critic_grad = critic_tape.gradient(critic_loss, self.critic.trainable_variables)       
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        with tf.GradientTape() as actor_tape:
            actor_actions =  self.actor(states)
            
            actor_loss = self.actor.loss(actions, actor_actions, returns - predict_critic)
            
        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))     

            
    @tf.function()
    def actor_forward(self, states):
        return self.actor(states)
    
    @tf.function()
    def critic_forward(self, states):
        return self.critic(states)
         
    
class Trainer:
    
    def __init__(self, agent: PAAC,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = np.argmax(self.agent.actor_forward(np.asarray(state, dtype=np.float32).reshape(1, -1))[0])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        j = 0
        
        venv = gym.vector.make(self.args.env, self.args.num_envs, asynchronous=True)
        states = venv.reset()[0]
        
        while True:
            for _ in range(self.args.evaluate_each if self.args.evaluate_each != None else 50):
                
                actions = tf.random.categorical(                                   
                                            tf.math.log(
                                                self.agent.actor_forward(
                                                    np.asarray(states, dtype=np.float32).reshape(self.args.num_envs, -1)
                                                    )
                                                ),
                                                num_samples=1
                                            )
                next_states, rewards, terminated, truncated, _ = venv.step(actions[:,0].numpy())
                dones = terminated | truncated
                
                
                returns = rewards[:,None] + self.args.gamma * self.agent.critic_forward(np.asarray(next_states, dtype=np.float32).reshape(self.args.num_envs, -1)) * np.logical_not(dones)[:,None]
                self.agent.train(states.astype(np.float32), actions, returns) 
                    
                states = next_states
                
                j+= 1
                      
            if self.args.evaluate_each != None:
                self.evaluate(j)
            
def main(env, args):
    model = PAAC(args, env)
    
    trainer = Trainer(model, env, args)
    trainer.train()
