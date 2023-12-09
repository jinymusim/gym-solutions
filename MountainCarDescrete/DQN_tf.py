
import numpy as np
import tensorflow as tf
import gymnasium as gym
import collections
import random
from discrete_env_wrapper import DiscreteEnv

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
parser.add_argument("--target_update_freq", default=10, type=int, help="Target update frequency")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--min_buffer", default=100, type=int, help="Training episodes")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiling per dimension")

class Agent:
    
    def __init__(self, env: gym.Env, args: argparse.Namespace) -> None:
        
        self.observation_space = env.observation_space.n
        self.action_space = env.action_space.n
        
        inputs = tf.keras.layers.Input(self.observation_space)
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(self.action_space, activation=None)(hidden)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss = tf.keras.losses.MeanSquaredError()
        )
        
        
    @tf.function
    def forward(self, states):
        return self.model(states)
    
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, returns, reward_mask):
        
        with tf.GradientTape() as tape:
            outputs = self.model(states)
            loss = self.model.compiled_loss(returns, outputs * reward_mask)
        grad = tape.gradient(loss, self.model.trainable_variables)
        
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
    @tf.function
    def copy_weights_from(self, other) -> None:
        for var, other_var in zip(self.model.variables, other.model.variables):
            var.assign(other_var)


class Trainer:
    
    def __init__(self, agent: Agent, target_agent : Agent,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.target_agent = target_agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action_value = self.agent.forward(np.asarray(state).reshape(1, -1))
                action = np.argmax(action_value.numpy()[0])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        
        replay_buffer = collections.deque()
        
        epsilon = self.args.epsilon
        j = 0
        for epoch in range(self.args.epochs):
            state, done = self.env.reset()[0], False
            while not done:
                
                if random.random() < epsilon:
                    action = random.choice(list(range(self.env.action_space.n)))
                else:
                    action = np.argmax(self.agent.forward(np.asarray(state, dtype=np.float32).reshape(1, -1)).numpy()[0])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                replay_buffer.append((state, action, reward, done, next_state))
                
                if len(replay_buffer) >= self.args.min_buffer:
                    if len(replay_buffer) % self.args.target_update_freq == 0:
                        self.target_agent.copy_weights_from(self.agent)

                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    returns = [reward + self.args.gamma*np.max(
                        self.target_agent.forward(np.asarray(next_state, dtype=np.float32).reshape(1, -1)).numpy()[0]
                    )*(1-done) for _, _, reward, done, next_state in episode]
                    returns = np.asarray(returns, dtype=np.float32)
                    mask = np.zeros(shape=(len(returns), self.env.action_space.n), dtype=np.float32)
                    for i in range(len(returns)):
                        mask[i,episode[i][1]] = 1
                        
                    self.agent.train(np.asarray([state for state,_,_,_,_ in episode], dtype=np.float32).reshape(self.args.batch_size, -1), returns[:, np.newaxis] * mask, mask)

                state = next_state
                
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            if self.args.epsilon_final_at:
                epsilon = np.interp(epoch + 1, [0, self.args.epsilon_final_at], [self.args.epsilon, self.args.epsilon_final])

            
def main(env, args):
    model = Agent(env, args)
    target = Agent(env, args)
    
    trainer = Trainer(model, target, env, args)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make(args.env)
    env = DiscreteEnv(env, args.tiles)
    
    main(env, args)
        