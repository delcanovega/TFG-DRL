import numpy as np
import random
import math

from keras.layers import Dense
from keras.models import Sequential

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

        # Neural Network structure
        # TODO: review (code from last year)
        self.model = Sequential()
        self.model.add(Dense(units=10, input_dim=DIMENSION, activation='relu'))
        self.model.add(Dense(units=action_space, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)

        # Define the loss function and the optimizer used to minimize it
        self.model.compile(loss='mse', optimizer=self.optimizer)

        # TODO: create a memory

    def predict_action(self, state):
        if random.uniform(0, 1) < EXPLORATION:
            action = random.randint(0, self.action_space - 1)  # Explore action space
        else:
            # TODO: predict using NN
            action = 0

        return action

    def update_memory(self, state):
        # TODO: update agent's memory
        

    # TODO: add get_minibatch ad replay if necessary