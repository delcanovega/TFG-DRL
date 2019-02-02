import random
import numpy as np

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

from collections import deque

MEMORY_SIZE = 2000
EXPLORATION_MIN = 0.1

class DQNAgent:
    def __init__(self, state_space, action_space, learning_rate, discount_factor, exploration):
        self.state_space = state_space
        self.action_space = action_space

        # Memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration = exploration

        self.epsilon_min = 0.01

        # Neural Network structure
        self.model = Sequential()
        # Input Layer of state size(4) and Hidden Layer with 16 nodes
        self.model.add(Dense(16, input_dim=self.state_space, activation='relu'))
        # Hidden layer with 16 nodes
        self.model.add(Dense(units=16, activation='linear'))
        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(self.action_space, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def predict_action(self, state):
        if random.uniform(0, 1) < self.exploration:
            # Explore
            action = random.randrange(self.action_space)
        else:
            # Exploit
            action_value = self.model.predict(np.array([state]))[0]  # Best action
            action = np.argmax(action_value)

        return action

    def update_table(self, observation):
        self.memory.append(observation)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.exploration > EXPLORATION_MIN:
            self.exploration *= 0.99

    # TODO: add get_minibatch ad replay if necessary
