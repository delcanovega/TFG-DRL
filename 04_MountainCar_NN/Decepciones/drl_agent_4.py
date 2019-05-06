import random
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import Adadelta
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

# RandomBatchAgentTwoBrains:
BESTOF = 100 # Number of simulations of the comparison

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

        #model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #probamos con este otro metodo
        model.compile(loss='mse', optimizer=Adadelta(lr=1.0, rho=self.discount_factor, epsilon=None, decay=0.0))

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


class RandomBatchAgent(DQNAgent):
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)

        self.memory = deque(maxlen=MEMORY_SIZE)

    def update_model(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def supports_replay():
        return True

    def save(self, name):
        self.model.save(name)

    def load(self,path):
       self.model = load_model(path)

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


class RandomBatchAgentTwoBrains(DQNAgent):
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)
        self.apprentice_model = self.build_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.firstTime = True

    def update_model(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def supports_replay():
        return True

    def save(self, name):
        self.model.save(name)

    def load(self,path):
        self.model = load_model(path)


    def replay(self):
        batch_size = min(len(self.memory), MINIBATCH_SIZE)
        minibatch = random.sample(self.memory, batch_size)

        # inputs = np.zeros((batch_size, self.state_space))
        # targets = np.zeros((batch_size, self.action_space))
        for state, action, reward, next_state, done in minibatch:
            q_update = -reward if done else self.bellman(reward, next_state)
            target = self.apprentice_model.predict(state)
            target[0][action] = q_update

            self.apprentice_model.fit(state, target, verbose=0)
            # np.append(inputs, state)
            # np.append(targets, target)
        # self.model.train_on_batch(inputs, targets)

    def replay_Mentor(self):
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

    def test(self, env, model):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]]) 

        done = False
        acc_reward = 0
        while not done:
            action_value = self.model.predict(state)
            action = np.argmax(action_value) 

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

            state = next_state
            acc_reward += reward
            
        return acc_reward

    def compete(self, env):
        """Compares the mentor and the apprentice returning the one that performs better"""

        if self.firstTime :
            self.apprentice_model.from_config(self.model.get_config())
            self.firstTime = False
            return -1
        
        total_reward_mentor = 0
        total_reward_apprentice = 0
        for _ in range(BESTOF):
            reward_mentor = self.test(env, self.model)
            reward_apprentice = self.test(env, self.apprentice_model)

            # print("Round {}: mentor {} - apprentice {}".format(i + 1, reward_mentor, reward_apprentice))
            total_reward_mentor += reward_mentor
            total_reward_apprentice += reward_apprentice

        #print("modelo Mentor\n",self.model.summary())
        #print("modelo Aprendiz\n", self.apprentice_model.summary())
        print("Final results: mentor {} - apprentice {}".format(total_reward_mentor, total_reward_apprentice))
        
        if total_reward_mentor > total_reward_apprentice :
            self.apprentice_model.set_weights(self.model.get_weights())
            return 0
        else : 
            self.model.set_weights(self.apprentice_model.get_weights())
            #self.model.from_config(self.apprentice_model.get_config())
            return 1



class RandomBatchAgentTwoBrainsBestSave(DQNAgent):
    def __init__(self, state_space, action_space):
        DQNAgent.__init__(self, state_space, action_space)
        self.apprentice_model = self.build_model()
        self.best_model = self.model
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.firstTime = True
        self.secondTime = True

    def update_model(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def supports_replay():
        return True

    def save(self, name):
        self.model.save(name)

    def load(self,path):
        self.model = load_model(path)


    def replay(self):
        batch_size = min(len(self.memory), MINIBATCH_SIZE)
        minibatch = random.sample(self.memory, batch_size)

        # inputs = np.zeros((batch_size, self.state_space))
        # targets = np.zeros((batch_size, self.action_space))
        for state, action, reward, next_state, done in minibatch:
            q_update = -reward if done else self.bellman(reward, next_state)
            target = self.apprentice_model.predict(state)
            target[0][action] = q_update

            self.apprentice_model.fit(state, target, verbose=0)
            # np.append(inputs, state)
            # np.append(targets, target)
        # self.model.train_on_batch(inputs, targets)

    def replay_Mentor(self):
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

    def test(self, env, model):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]]) 

        done = False
        acc_reward = 0
        while not done:
            action_value = self.model.predict(state)
            action = np.argmax(action_value) 

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

            state = next_state
            acc_reward += reward
            
        return acc_reward

    def compete(self, env):
        """Compares the mentor and the apprentice returning the one that performs better"""

        if self.secondTime:
            if self.firstTime :
                self.apprentice_model.from_config(self.model.get_config())
                self.firstTime = False
                return -1
            self.secondTime = False
            return -1
        
        total_reward_mentor = 0
        total_reward_apprentice = 0
        for _ in range(BESTOF):
            reward_mentor = self.test(env, self.model)
            reward_apprentice = self.test(env, self.apprentice_model)

            # print("Round {}: mentor {} - apprentice {}".format(i + 1, reward_mentor, reward_apprentice))
            total_reward_mentor += reward_mentor
            total_reward_apprentice += reward_apprentice

        #print("modelo Mentor\n",self.model.summary())
        #print("modelo Aprendiz\n", self.apprentice_model.summary())
        print("Final results: mentor {} - apprentice {}".format(total_reward_mentor, total_reward_apprentice))
        
        if total_reward_mentor > total_reward_apprentice :
            self.apprentice_model.set_weights(self.model.get_weights())
            return 0
        else : 
            self.model.set_weights(self.apprentice_model.get_weights())
            #self.model.from_config(self.apprentice_model.get_config())
            return 1

    def bigCompete(self, env):
        """Compares the mentor, the apprentice and the best model and save the best"""
        if self.secondTime:
            if self.firstTime :
                self.apprentice_model.from_config(self.model.get_config())
                self.firstTime = False
                return -1, -1
            self.secondTime = False
            return -1, -1
        
        total_reward_mentor = 0
        total_reward_apprentice = 0
        total_reward_max = 0
        for _ in range(BESTOF):
            reward_mentor = self.test(env, self.model)
            reward_apprentice = self.test(env, self.apprentice_model)
            reward_max = self.test(env, self.best_model)

            # print("Round {}: mentor {} - apprentice {}".format(i + 1, reward_mentor, reward_apprentice))
            total_reward_mentor += reward_mentor
            total_reward_apprentice += reward_apprentice
            total_reward_max += reward_max

        #print("modelo Mentor\n",self.model.summary())
        #print("modelo Aprendiz\n", self.apprentice_model.summary())
        print("Final results: mentor {} - apprentice {} - best player {}".format(total_reward_mentor, total_reward_apprentice, total_reward_max))
        b = 2
        if total_reward_mentor > total_reward_apprentice :
            self.apprentice_model.set_weights(self.model.get_weights())
            a = 0
            if total_reward_mentor > total_reward_max:
                self.best_model.set_weights(self.model.get_weights())
                b = 0
            else:
                self.model.set_weights(self.best_model.get_weights())
        elif total_reward_apprentice >= total_reward_mentor : 
            a = 1
            if total_reward_apprentice > total_reward_max:
                self.model.set_weights(self.apprentice_model.get_weights())
                self.best_model.set_weights(self.apprentice_model.get_weights())
                b = 1
            else:
                self.model.set_weights(self.best_model.get_weights())
         
        return a, b



