import tensorflow as tf
import numpy as np
import gymnasium as gym
import argparse



class AlphaZeroAgent:
    
    class Node:
        def __init__(self, env: gym.Env, prob:float) -> None:
            self.prob = prob
            self.state = None
            self.env = env
            self.children = {}
            self._visits = 0
            self._value = 0
            
        def value(self):
            return -1 if self._visits ==0 else self._value/self._visits
    
        def is_evaluated(self):
            return self._visits > 0
        
        def eval(self, agent, state: np.ndarray, done, reward):
            self.state = state
            if done:
                value = reward
            else:
                actions, value = agent.predict(state.reshape(1, -1))
                value = value[0]
                for i in range(self.env.action_space.n):
                    self.children[i] = AlphaZeroAgent.Node(self.env, actions[0,i])
            self._value, self._visits = value, 1
                

    
    class Network(tf.keras.Model):
        def __init__(self, args: argparse.Namespace, env: gym.Env, *kwargs):
            super().__init__(*kwargs)
            self.hidden = tf.keras.layers.Dense(args.hiden_size, activation="relu")
            self.value = tf.keras.layers.Dense(1, activation=None)
            self.action = tf.keras.layers.Dense(env.action_space.n, activation="softmax")
            
        def call(self,inputs: tf.Tensor):
            hidden = self.hidden(inputs)
            value = self.value(hidden)
            action = self.action(hidden)
            return action, value
            
    
    def __init__(self, args: argparse.Namespace, env: gym.Env) -> None:
        self.model = AlphaZeroAgent.Network(args, env)
        self.model.compile(
            optimizer="adam",
            loss=[tf.losses.CategoricalCrossentropy(), tf.losses.MeanSquaredError()]
        )

    @tf.function(experimental_relax_shapes = True)
    def train(self, states, actions, values):
        with tf.GradientTape() as tape:
            predict_action, predict_value = self.model(states)
            action_loss = self.model.compiled_loss[0](actions, predict_action)
            value_loss = self.model.compiled_loss[1](values, predict_value)
        grad = tape.gradient([action_loss, value_loss], self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    @tf.function()
    def predict(self, states):
        return self.model(states)

