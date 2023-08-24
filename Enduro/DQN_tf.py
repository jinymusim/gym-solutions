
import random
import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import gymnasium as gym
import numpy as np
import collections

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="ALE/Enduro-v5", type=str, help="Environment.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--num_filters", default=[8,8,8], type=list, help="Number of filters for convolution")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size")
parser.add_argument("--target_forgetting", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--min_buffer", default=256, type=int, help="Training episodes")
parser.add_argument("--frames", default=3, type=int, help="number of frames to stack")
parser.add_argument("--num_pixels", default=64, type=int, help="number of frames to stack")


class Network:
    
        
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        
        inputs = tf.keras.layers.Input(env.observation_space.shape)
        hidden = inputs
        for filt in args.num_filters:
            hidden = tf.keras.layers.Conv2D(filt, kernel_size=3, strides=2, activation="relu", input_shape=env.observation_space.shape[1:])(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")(hidden)
        advantage = tf.keras.layers.Dense(env.action_space.n)(hidden)
        value = tf.keras.layers.Dense(1)(hidden)
        outputs = advantage + value  - tf.reduce_mean(advantage)
        
        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)
        self.model.compile(
            optimizer="adam",
            loss = "mse"
        )
        
        self.target_model = tf.keras.models.clone_model(self.model)
        
        self.target_forgetting = args.target_forgetting


        
    @tf.function(experimental_relax_shapes=True)    
    def train(self, states, q_values, mask):
        with tf.GradientTape() as tape:
            model_pred = self.model(states)
            loss = self.model.compiled_loss(q_values, model_pred * mask)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
        
        for var, target_var in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            target_var.assign(target_var * (1 - self.target_forgetting) + var * self.target_forgetting)
        
            
    @tf.function()
    def model_predict(self, states):
        return self.model(states)
    
    @tf.function()
    def target_predict(self, states):
        return self.target_model(states)
         
    
class Trainer:
    
    def __init__(self, agent: Network,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = np.argmax(self.agent.model_predict(np.asarray(state).reshape(1, *self.env.observation_space.shape)).numpy()[0])
                    
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
                action = np.argmax(self.agent.model_predict(np.asarray(state).reshape(1, *self.env.observation_space.shape)).numpy()[0])
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                replay_buffer.append((state, action, reward, done, next_state))

                state = next_state
                
                if len(replay_buffer) >= self.args.min_buffer:
                    episode = random.choices(replay_buffer, k=self.args.batch_size)
                    returns = [reward + self.args.gamma * np.max(
                        self.agent.target_predict(np.asarray(next_state).reshape(1, *self.env.observation_space.shape)).numpy()[0]
                    )*(1-done) for _, _, reward, done, next_state in episode 
                               ]
                    returns = np.asarray(returns)
                    mask = np.zeros(shape=(self.args.batch_size, self.env.action_space.n), dtype=np.float32)
                    for i in range(self.args.batch_size):
                        mask[i,episode[i][1]] = 1
                    self.agent.train(np.asarray([np.asarray(state) for state,_,_,_,_ in episode]), returns, mask)  

            
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = Network(args, env)
    
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
