import numpy as np
import random
import math

# Hyperparameters
LEARNING_RATE = 0.1     # Alpha
DISCOUNT_FACTOR = 1.0   # Gamma
EXPLORATION = 0.5       # Epsilon (initial)
MIN_EXPLORATION = 0.01  # Epsilon (final)


class QLAgent:
    def __init__(self, config, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

        # Bounds: [cart_position, cart_velocity, pole_angle, pole_velocity]
        self.upper_bounds = [self.state_space.high[0], 0.5, self.state_space.high[2], math.radians(50)]
        self.lower_bounds = [self.state_space.low[0], -0.5, self.state_space.low[2], -math.radians(50)]

        self.exploration = EXPLORATION

        self.buckets = config
        self.q_table = np.zeros(self.buckets + (self.action_space,))

    def discretize(self, obs):
        ratios = [(obs[i] + self.upper_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(obs))]
        classified_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        classified_obs = [min(self.buckets[i] - 1, max(0, classified_obs[i])) for i in range(len(obs))]

        return tuple(classified_obs)

    def predict_action(self, state):
        discrete_state = self.discretize(state)

        if random.uniform(0, 1) < self.exploration:
            action = random.randint(0, self.action_space - 1)  # Explore action space
        else:
            action = np.argmax(self.q_table[discrete_state])  # Exploit learned values

        return action

    def update_table(self, old_state, next_state, action, reward):
        discrete_old_state = self.discretize(old_state)
        discrete_next_state = self.discretize(next_state)

        old_value = self.q_table[discrete_old_state][action]
        next_max = np.max(self.q_table[discrete_next_state])

        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        self.q_table[discrete_old_state][action] = new_value

    def decrease_exploration(self):
        if self.exploration > MIN_EXPLORATION:
            self.exploration *= 0.99
