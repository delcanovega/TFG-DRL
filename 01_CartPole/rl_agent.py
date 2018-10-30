import numpy as np
import random
import math

# Hyperparameters
# TODO: improve - maybe gridsearch-like method?
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0
EXPLORATION = 0.35

class QLAgent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.upper_bounds = [self.state_space.high[0], 0.5, self.state_space.high[2], math.radians(50)]
        self.lower_bounds = [self.state_space.low[0], -0.5, self.state_space.low[2], -math.radians(50)]

        self.buckets = (1, 1, 6, 12)  # TODO: try out other values
        self.n_states = 1
        for i in self.buckets:
            self.n_states *= i

        self.q_table = np.zeros([self.n_states, self.action_space])

    def discretize(self, obs):
        ratios = [(obs[i] + self.upper_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(obs))]
        classified_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        classified_obs = [min(self.buckets[i] - 1, max(0, classified_obs[i])) for i in range(len(obs))]

        state = 0
        for i in range(len(classified_obs)):
            state += classified_obs[i] if i == 0 else (classified_obs[i]) * self.buckets[i-1]

        return state

    def predict_action(self, state):
        discrete_state = self.discretize(state)

        if random.uniform(0, 1) < EXPLORATION:
            action = random.randint(0, self.action_space - 1)  # Explore action space
        else:
            action = np.argmax(self.q_table[discrete_state])  # Exploit learned values

        return action

    def update_table(self, old_state, next_state, action, reward):
        discrete_old_state = self.discretize(old_state)
        discrete_next_state = self.discretize(next_state)

        old_value = self.q_table[discrete_old_state, action]
        next_max = np.max(self.q_table[discrete_next_state])

        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        self.q_table[discrete_old_state, action] = new_value

        # TODO: reduce the learning rate and exploration
