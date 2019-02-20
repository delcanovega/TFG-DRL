import random
import numpy as np

from keras.models     import Sequential
from keras.layers     import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

MEMORY_SIZE = 2000
HIDDEN_LAYER_SIZE = 12

# HYPERPARAMETERS
LEARNING_RATE = 0.1   # Alpha (try higher values for minibatch training)
DISCOUNT_FACTOR = 0.95  # Gamma
EXPLORATION = 0.5       # Epsilon (initial)
MIN_EXPLORATION = 0.01  # Epsilon (final)


# TODO: 
# 1. entrenar por minibatch
# 2. regularizar red
# 3. probar red mas pequeña

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
        self.memory = deque(maxlen=MEMORY_SIZE)
    
    def build_model(self):
        # Neural Network structure
        model = Sequential()

        model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=self.state_space, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu'))
        # model.add(Dropout(0.2))
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

    def update_model (self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def getMinibatch (self, minibatch_size):
        indexes = np.random.choice(np.arange(len(self.memory)), size=minibatch_size, replace=False)
        minibatch = []
        for index in indexes:
            minibatch.append(self.memory[index])
        
        return minibatch

    def replay(self, minibatch_size):
        
        if(len(self.memory) > minibatch_size):
            minibatch = self.getMinibatch(minibatch_size)

            inputs = np.zeros((minibatch_size,self.state_space))
            targets = np.zeros((minibatch_size, self.action_space))

            i=0
            for state, action, reward, next_state, done in minibatch:
                
                inputs[i] = state
                targets[i] = self.model.predict(state)[0]
                
                if done:
                    targets[i, action] = reward
                else:
                    targets[i, action] = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            
                i+=1

            # self.model.fit(inputs, targets, batch_size=minibatch_size, epochs=1, verbose=0)
            self.model.train_on_batch(inputs, targets)

    def update_hyperparameters(self):
        if self.exploration > MIN_EXPLORATION:
            self.exploration = self.exploration * 0.98

class SimpleAgent(DQNAgent):
    """Simplest implementation of an agent using a Neural Network.

    It is a direct replacement of the Q-table from the RLAgent.

    """
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)

    def update_model(self, state, action, reward, next_state):
        old_value = self.model.predict(state)
        next_max = np.max(self.model.predict(next_state))

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)

        self.model.fit(state, new_value, epochs=1, verbose=0)


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
