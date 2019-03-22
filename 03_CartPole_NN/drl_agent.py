import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.initializers import Constant

from collections import deque


# HYPERPARAMETERS

# Reinforcement Learning:
LEARNING_RATE = 0.0001      # Alpha (try higher values for minibatch training)
DISCOUNT_FACTOR = 0.95      # Gamma
EXPLORATION = 0.5           # Epsilon (initial)
MIN_EXPLORATION = 0.01      # Epsilon (final)

# Agent:
MEMORY_SIZE = 2000
HIDDEN_LAYER_SIZE = 24
MINIBATCH_SIZE = 32


class DQNAgent:
    """Base class for the different agent implementations.

    Note:
        Don't instantiate this class directly

    """
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        # Hyperparameters
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration = EXPLORATION

        self.model = self.build_model()
    
    def build_model(self):
        # Neural Network structure
        model = Sequential()

        model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=self.state_space, activation='relu'))
        model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_space, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def predict_action(self, state):
        if random.uniform(0, 1) < self.exploration:
            action = random.randrange(self.action_space)  # Explore
        else:
            action_value = self.model.predict(state)
            action = np.argmax(action_value)  # Exploit
        return action
    
    def bellman(self, reward, next_state):
        return reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])

    def update_hyperparameters(self):
        if self.exploration > MIN_EXPLORATION:
            self.exploration *= 0.995


class SimpleAgent(DQNAgent):
    """Simplest implementation of an agent using a Neural Network.

    It is a direct replacement of the Q-table from the RLAgent.

    """
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)

    def update_model(self, state, action, reward, next_state, done):
        q_update = -reward if done else self.bellman(reward, next_state)
        target = self.model.predict(state)
        target[0][action] = q_update

        self.model.fit(state, target, epochs=1, verbose=0)

    @staticmethod
    def supports_replay():
        return False


class BatchAgent(DQNAgent):
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)

        self.memory = deque(maxlen=MINIBATCH_SIZE)

    def update_model(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def supports_replay():
        return True

    def replay(self):
        iter = reversed(self.memory)
        batch_size = min(len(self.memory), MINIBATCH_SIZE)

        #inputs = np.empty((batch_size, self.state_space))
        #targets = np.empty((batch_size, self.action_space))
        for _ in range(batch_size):
            state, action, reward, next_state, done = next(iter)

            q_update = -reward if done else self.bellman(reward, next_state)
            target = self.model.predict(state)
            target[0][action] = q_update

            self.model.fit(state, target, verbose=0)

            #np.append(inputs, state)
            #np.append(targets, target)

        #self.model.train_on_batch(inputs, targets)


class RandomBatchAgent(DQNAgent):
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)

        self.memory = deque(maxlen=MEMORY_SIZE)

    def update_model(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def supports_replay():
        return True

    def replay(self):
        batch_size = min(len(self.memory), MINIBATCH_SIZE)
        minibatch = random.sample(self.memory, batch_size)

        # inputs = np.zeros((batch_size, self.state_space))
        # targets = np.zeros((batch_size, self.action_space))
        for state, action, reward, next_state, done in minibatch:
            q_update = -reward if done else self.bellman(reward, next_state)
            target = self.model.predict(state)
            target[0][action] = q_update

            self.model.fit(state, target, verbose=0)
            # np.append(inputs, state)
            # np.append(targets, target)
        # self.model.train_on_batch(inputs, targets)
