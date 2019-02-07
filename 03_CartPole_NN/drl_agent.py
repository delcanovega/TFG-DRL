import random
import numpy as np

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

from collections import deque

MEMORY_SIZE = 2000
EXPLORATION_MIN = 0.1
HIDDEN_LAYER_SIZE = 16 # probar red mas pequeña

# HYPERPARAMETERS
LEARNING_RATE = 0.001 # subir un poco para minibatch
DISCOUNT_FACTOR = 0.95
EXPLORATION = 1.0

# TODO: 
# 1. entrenar por minibatch
# 2. regularizar red
# 3. probar red mas pequeña

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
        
        if self.exploration > EXPLORATION_MIN:
            self.exploration *= 0.995   # TODO: no va aqui si no en el main, o funcion aparte decrease_exploration
