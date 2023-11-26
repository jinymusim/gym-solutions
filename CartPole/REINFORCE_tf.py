
import numpy as np
import tensorflow as tf
import gymnasium as gym

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layers sizes")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs to train for")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("--evaluate_each", default=50, type=int, help="After how many epochs to evaluate")
parser.add_argument("--evaluate_for", default=10, type=int, help="For How many Epochs to evaluate")


class Agent:
    
    def __init__(self, env: gym.Env, args: argparse.Namespace) -> None:
        
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        
        inputs = tf.keras.layers.Input(self.observation_space)
        
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(self.action_space, activation="softmax")(hidden)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        )
        
    @tf.function
    def forward(self, states):
        return self.model(states)
    
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            outputs = self.model(states)
            loss = self.model.compiled_loss(actions, outputs, returns)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
class Trainer:
    
    def __init__(self, agent: Agent, env: gym.Env, epochs: int, batch_size:int, evaluate_each:int = None, evaluate_for:int = 10) -> None:
        self.agent = agent
        self.env = env
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluate_each = evaluate_each
        self.evaluate_for = evaluate_for
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                agent_prob = self.agent.forward(np.asarray(state).reshape(1, -1))
                action = np.argmax(agent_prob.numpy()[0,:])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
    def train(self):
        j = 0
        for _ in range(self.epochs):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(self.batch_size):
                states, actions, rewards = [], [], []
                state, done = self.env.reset()[0], False
                while not done:
                    agent_prob = self.agent.forward(np.asarray(state, dtype=np.float32).reshape(1, -1))
                    action = np.random.choice(list(range(self.env.action_space.n)), p=agent_prob.numpy()[0,:])
                    
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
            
            self.agent.train(np.asarray(batch_states, dtype=np.float32).reshape(-1, self.env.observation_space.shape[0]), 
                             np.asarray(batch_actions, dtype=np.int32),
                             np.asarray(batch_returns, dtype=np.float32))
                
                      
            if self.evaluate_each != None and j % self.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = Agent(env, args)
    
    trainer = Trainer(model, env, args.epochs, args.batch_size, args.evaluate_each, args.evaluate_for)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make("CartPole-v1")
    
    main(env, args)
        