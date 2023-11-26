
import numpy as np
import tensorflow as tf
import gymnasium as gym
import argparse

class Agent:
    
    class Actor(tf.keras.Model):
        def __init__(self, args: argparse.Namespace, env : gym.Env, *kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hidden_layer, activation="relu")
            self.outputs = tf.keras.layers.Dense(env.action_space.n, activation="softmax")
        
        
        def call(self, inputs: tf.Tensor):
            hidden = self.hidden(inputs)
            outputs = self.outputs(hidden)
            return outputs
        
    class Baseline(tf.keras.Model):
        def __init__(self, args: argparse.Namespace, env: gym.Env, *kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hidden_layer, activation="relu")
            self.baseline = tf.keras.layers.Dense(1, activation=None)
            
        def call(self, inputs: tf.Tensor):
            hidden = self.hidden(inputs)
            baseline = self.baseline(hidden)
            return baseline
    
    def __init__(self, env: gym.Env, args: argparse.Namespace) -> None:
        
        self.actor = Agent.Actor(args, env)
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        )
        
        self.baseline = Agent.Baseline(args, env)
        self.baseline.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss = tf.keras.losses.MeanSquaredError()
        )
        
        
    @tf.function
    def forward(self, states):
        return self.actor(states)
    
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        
        with tf.GradientTape() as baseline_tape:
            baseline_outputs = self.baseline(states)
            baseline_loss = self.baseline.compiled_loss(returns, baseline_outputs)
        baseline_grad = baseline_tape.gradient(baseline_loss, self.baseline.trainable_variables)
        
        with tf.GradientTape() as tape:
            outputs = self.actor(states)
            loss = self.actor.compiled_loss(actions, outputs, returns - tf.squeeze(baseline_outputs) )
        grad = tape.gradient(loss, self.actor.trainable_variables)
        
        self.actor.optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))
        self.baseline.optimizer.apply_gradients(zip(baseline_grad, self.baseline.trainable_variables))
    
class Trainer:
    
    def __init__(self, agent: Agent, env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
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
        for _ in range(self.args.epochs):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(self.args.batch_size):
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
            
            self.agent.train(np.asarray(batch_states, dtype=np.float32).reshape(-1, self.env.observation_space.n), 
                             np.asarray(batch_actions, dtype=np.int32),
                             np.asarray(batch_returns, dtype=np.float32))
                
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
            j+= 1
            
def main(env, args):
    model = Agent(env, args)
    
    trainer = Trainer(model, env, args)
    trainer.train()
    
        