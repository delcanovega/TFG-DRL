import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.initializers import Constant

from collections import deque


# HYPERPARAMETERS

# Reinforcement Learning:
LEARNING_RATE = 0.001         # Alpha (try higher values for minibatch training)
DISCOUNT_FACTOR = 0.95      # Gamma
EXPLORATION = 1.0           # Epsilon (initial)
MIN_EXPLORATION = 0.01      # Epsilon (final)

# Agent:
MEMORY_SIZE = 2000
HIDDEN_LAYER_SIZE = 24
MINIBATCH_SIZE = 20


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
        return reward + self.learning_rate * np.amax(self.model.predict(next_state)[0])

    def update_hyperparameters(self):
        if self.exploration > MIN_EXPLORATION:
            self.exploration = self.exploration * 0.995


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

    def get_minibatch(self, batch_size):
        indexes = np.random.choice(np.arange(len(self.memory)), size=batch_size, replace=False)
        # minibatch = []
        # for index in indexes:
        #     minibatch.append(self.memory[index])

        return indexes

    def replay(self):
        if len(self.memory) < MINIBATCH_SIZE:
            batch_size = len(self.memory)
        else:
            batch_size = MINIBATCH_SIZE    
            
        # minibatch = self.get_minibatch(batch_size)

        inputs = np.zeros((len(self.memory), self.state_space))
        targets = np.zeros((len(self.memory), self.action_space))

        i = len(self.memory)
        for state, action, reward, next_state, done in self.memory:

            inputs[i-1] = state
            targets[i-1] = self.model.predict(state)[0]
            targets[i-1, action] = reward

            if not done:
                targets[i-1, action] = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])

            i -= 1
        inputs_batch = np.zeros((batch_size, self.state_space))
        targets_batch = np.zeros((batch_size, self.action_space))

        minibatch = self.get_minibatch(batch_size)
        j=0
        for index in minibatch:
            inputs_batch[j] = inputs[index]
            targets_batch[j] = targets[index]
            j+=1
        self.model.train_on_batch(inputs_batch, targets_batch)


# Deprecated agent:
# keep here as a reference until it is migrated and then delete
class DeprecatedDQNAgent:
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
        self.model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=self.state_space, activation='relu'))
        self.model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu')) # sigmoid # dropout # regularizacion L1 o L2
        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(self.action_space, activation='linear')) #softmax

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
        # actualizar valores antes de añadirlos a la memoria

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]) # <--- TODO: REVISAR
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # <--- TODO: fit de minibatch entero, no de 1 en 1
        
        if self.exploration > MIN_EXPLORATION:
            self.exploration *= 0.995   # TODO: no va aqui si no en el main, o funcion aparte decrease_exploration
