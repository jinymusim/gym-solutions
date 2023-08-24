import argparse
import tensorflow as tf
import gymnasium as gym
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--hidden_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")

parser.add_argument("--num_envs", default=8, type=int, help="Environments.")
parser.add_argument("--worker_steps", default=250, type=int, help="Steps for each worker to perform.")

parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")

parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")


class PPO:
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        inputs = tf.keras.layers.Input(env.observation_space.shape)
        hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")(inputs)
        action = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)(hidden)
        
        hidden = tf.keras.layers.Dense(args.hidden_size, activation="relu")(inputs)
        value = tf.keras.layers.Dense(1, activation=None)(hidden)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=[action, value])
        
        self.model.compile(
            optimizer="adam"
        )
        self.args  = args
    
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, mask ,action_probs, advantages, returns):
        with tf.GradientTape() as tape:
            action, value = self.model(states)
            policy_comp = tf.reduce_sum(action * mask, axis=-1)/action_probs
            ppo_loss = -1 * tf.reduce_mean(tf.minimum(policy_comp * advantages, tf.clip_by_value(policy_comp, 1-self.args.clip_epsilon, 1+self.args.clip_epsilon)* advantages))
            mse_loss = tf.losses.mean_squared_error(value, returns)
            reg_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(action, action)) * self.args.entropy_regularization
            loss = mse_loss + ppo_loss - reg_loss
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
    @tf.function()
    def forward(self, states):
        return self.model(states)
    
    
class Trainer:
    
    def __init__(self, agent: PPO,env: gym.Env, args: argparse.Namespace) -> None:
        self.agent = agent
        self.env = env
        self.args = args
        
    def evaluate(self, j):
        rewards_count = []
        for _ in range(self.args.evaluate_for):
            state, done = self.env.reset()[0], False
            total_reward = 0
            while not done:
                action = np.argmax(self.agent.forward(np.asarray(state, dtype=np.float32).reshape(1, -1))[0])
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                state = next_state
            rewards_count.append(total_reward)
        print(f"Evaluation After {j} epochs: {np.mean(rewards_count)} +- {np.std(rewards_count, ddof=1)}")
        
        
    def train(self):
        j = 0
        
        venv = gym.vector.make(self.args.env, self.args.num_envs, asynchronous=True)
        state = venv.reset()[0]
        
        while True:
            states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
            for _ in range(self.args.worker_steps):
                
                policy, value = self.agent.forward(state.reshape(self.args.num_envs, -1))
                policy = policy.numpy()
                action = np.asarray([np.random.choice(self.env.action_space.n, size=1, p=policy[i,:])[0] for i in range(self.args.num_envs) ])
                action_prob = np.take(policy, action)
                action_mask = np.zeros_like(policy)
                np.put_along_axis(action_mask, action.reshape(-1,1 ), values=1, axis=1)
                next_state, reward, terminated, truncated, _ = venv.step(action)
                done = terminated | truncated
                
                states.append(state)
                actions.append(action_mask),
                action_probs.append(action_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value.numpy()[:,0])

                state = next_state
                
            advantages = np.zeros((self.args.worker_steps, self.args.num_envs))
            returns = np.zeros_like(advantages)
            for i in range(self.args.num_envs):
                k = self.args.worker_steps - 2
                gae = 0
                while k >= 0:
                    delta = rewards[k][i] + self.args.gamma * values[k+1][i] - values[k][i]
                    gae = delta + self.args.gamma * self.args.trace_lambda * gae
                    advantages[k,i] = gae
                    returns[k,i] = gae + values[k][i]
                    k-=1
                    
            for batch_state, batch_mask, batch_action_prob, batch_advantages, batch_returns in zip(
                np.array_split(np.concatenate(states),self.args.worker_steps *  self.args.num_envs // self.args.batch_size),
                np.array_split(np.concatenate(actions), self.args.worker_steps *  self.args.num_envs // self.args.batch_size),
                np.array_split(np.concatenate(action_probs), self.args.worker_steps *  self.args.num_envs // self.args.batch_size),
                np.array_split(np.concatenate(advantages), self.args.worker_steps *  self.args.num_envs // self.args.batch_size),
                np.array_split(np.concatenate(returns), self.args.worker_steps *  self.args.num_envs // self.args.batch_size)):
                self.agent.train(batch_state.astype(np.float32), batch_mask.astype(np.float32), 
                                 batch_action_prob.astype(np.float32), batch_advantages.astype(np.float32), batch_returns.astype(np.float32))
          
                        
            j+= 1
                      
            if self.args.evaluate_each != None and j % self.args.evaluate_each == 0:
                self.evaluate(j)
            
def main(env, args):
    model = PPO(args, env)
    
    trainer = Trainer(model, env, args)
    trainer.train()
    
if __name__ == "__main__": 
    args = parser.parse_args([] if "__file__" not in globals() else None)
    env = gym.make(args.env)
    
    main(env, args)
            
            