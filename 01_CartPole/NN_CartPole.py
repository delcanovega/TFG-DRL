import random
import numpy as np

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

from collections import deque

MEMORY_SIZE = 2000
EXPLORATION_MIN = 0.1
HIDDEN_LAYER_SIZE = 16

# HYPERPARAMETERS
LEARNING_RATE = 1.0
DISCOUNT_FACTOR = 0.95
EXPLORATION = 1.0

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        # Memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Hyperparameters
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration = EXPLORATION

        # Neural Network structure
        self.model = Sequential()
        # Input Layer of state size(4) and Hidden Layer with 16 nodes
        self.model.add(Dense(24, input_dim=self.state_space, activation='relu'))
        # Hidden layer with 16 nodes
        self.model.add(Dense(24, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(self.action_space, activation='linear'))

        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def predict_action(self, state):
        if np.random.rand() < self.exploration:
            # Explore
            action = random.randrange(self.action_space)
        else:
            # Exploit
            action_value = self.model.predict(state)
            action = np.argmax(action_value[0])
        return action

    def update_table(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.exploration > EXPLORATION_MIN:
            self.exploration *= 0.995

