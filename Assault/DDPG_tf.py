
import random
import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import gymnasium as gym
import numpy as np
import collections

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="ALE/Assault-v5", type=str, help="Environment.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--num_filters", default=[4,4,8], type=list, help="Number of filters for convolution")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_forgetting", default=0.001, type=float, help="Target network update weight.")
parser.add_argument("--min_buffer", default=64, type=int, help="Training episodes")
parser.add_argument("--ornstein_theta", default=0.15, type=float, help="Target network update weight.")
parser.add_argument("--ornstein_sigma", default=0.04, type=float, help="Target network update weight.")
parser.add_argument("--frames", default=3, type=int, help="number of frames to stack")
parser.add_argument("--num_pixels", default=64, type=int, help="number of frames to stack")



class DDPG:
    
    class Actor(tf.keras.Model):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.conv_layers = []
            for filt in args.num_filters:
                self.conv_layers.append(tf.keras.layers.Conv2D(filt, kernel_size=5, strides=3, activation="relu", input_shape=env.observation_space.shape[1:]))
            self.flat =  tf.keras.layers.Flatten()
            self.hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.actions  = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
            
        def call(self, inputs: tf.Tensor):
            hidden = inputs
            for layer in self.conv_layers:
                hidden = layer(hidden)
            hidden = self.flat(hidden)
            hidden = self.hidden(hidden)
            actions = self.actions(hidden)   
            
            
            return actions
        
    class Critic(tf.keras.Model):
        
        def __init__(self, args: argparse.Namespace, env: gym.Env,*kwargs):
            super().__init__(*kwargs)
            self.conv_layers = []
            for filt in args.num_filters:
                self.conv_layers.append(tf.keras.layers.Conv2D(filt, kernel_size=3, strides=2, activation="relu", input_shape=env.observation_space.shape[1:]))
            self.flat =  tf.keras.layers.Flatten()
            self.concat = tf.keras.layers.Concatenate()
            self.hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.hidden2 = tf.keras.layers.Dense(args.hidden_size, activation="relu")
            self.output_val = tf.keras.layers.Dense(1, activation=None)
            
        def call(self, inputs_img:tf.Tensor, inputs_action: tf.Tensor):
            hidden = inputs_img
            for layer in self.conv_layers:
                hidden = layer(hidden)
            hidden = self.flat(hidden)
            hidden = self.hidden(hidden)
            hidden2 =  self.hidden2(inputs_action)
            concat = self.concat([hidden, hidden2])
            outputs = self.output_val(concat)
            return outputs
        
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        self.actor = DDPG.Actor(args, env)
        self.actor.compile(optimizer= "adam")
        
        self.target_actor = DDPG.Actor(args,env)
        self.target_actor.compile()
        
        self.critic = DDPG.Critic(args, env)
        self.critic.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        self.target_critic = DDPG.Critic(args, env)
        self.target_critic.compile()
        
        self.target_forgetting = args.target_forgetting
        
        self.ornstein_process = np.zeros(env.action_space.n)
        self.ornstein_theta = args.ornstein_theta
        self.ornstein_sigma = args.ornstein_sigma


        
    @tf.function(experimental_relax_shapes=True)    
    def train(self, states, actions, returns):
        with tf.GradientTape() as actor_tape:
            actor_actions =  self.actor(states)
            
            critic_val = self.critic(states, actor_actions)
            
            actor_loss = -tf.reduce_mean(critic_val)
        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
        
        with tf.GradientTape() as critic_tape:          
            # Predicted values
            predict_critic = self.critic(states, actions)
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
        actor_actions = self.target_actor(states)
        return self.target_critic(states, actor_actions)
    
    def ornstein_noise(self):
        self.ornstein_process += self.ornstein_theta * (np.zeros_like(self.ornstein_process) - self.ornstein_process) + np.random.normal(scale=self.ornstein_sigma, size=self.ornstein_process.shape)
        return self.ornstein_process
         
    
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
                action = self.agent.actor_forward(np.asarray(state, dtype=np.float32).reshape(1, *self.env.observation_space.shape))[0]
                    
                next_state, reward, terminated, truncated, _ = self.env.step(np.argmax(action))
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
                action = self.agent.actor_forward(np.asarray(state, dtype=np.float32).reshape(1, *self.env.observation_space.shape)).numpy()[0] + self.agent.ornstein_noise()
                weight_cenetring = action -  action.min()
                selected_action = np.random.choice(self.env.action_space.n, 1, p=weight_cenetring/weight_cenetring.sum())[0]
                next_state, reward, terminated, truncated, _ = self.env.step(selected_action)
                done = terminated or truncated

                replay_buffer.append((state, weight_cenetring/weight_cenetring.sum(), reward, done, next_state))

                state = next_state
                
                if len(replay_buffer) >= self.args.min_buffer:
                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[snapshot for snapshot in episode]))
                    returns = rewards[:,None] + self.args.gamma * self.agent.critic_forward(np.asarray(next_states, dtype=np.float32).reshape(self.args.batch_size, *self.env.observation_space.shape)) * np.logical_not(dones)[:,None]
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
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.ResizeObservation(env, args.num_pixels)
    # The way framestack is done is stupid
    env = gym.wrappers.FrameStack(env,args.frames)
    
    main(env, args)
