import gym
import numpy as np
import random

# Hyperparameters
# TODO: improve
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.5
EXPLORATION = 0.5
EXPLORATION_MIN = 0.001

#
DISCRETIZED_STATES = 162

# Angle conversions
ONE_DEGREE = 0.0174532
SIX_DEGREES = 0.1047192
TWELVE_DEGREES = 0.2094384
FIFTY_DEGREES = 0.87266

# TODO: move this class to a different file
class QLAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        self.epsilon = EXPLORATION

        self.q_table = np.zeros([DISCRETIZED_STATES, action_space])

    @staticmethod
    def discretize(state):
        # FIXME: the -1 state is overwriting the state 161
        # TODO: find a different state representation
        cart_position = state[0]
        cart_velocity = state[1]
        pole_angle = state[2]
        pole_velocity = state[3]

        if not (-2.4 < cart_position < 2.4) or not (-TWELVE_DEGREES < pole_angle < TWELVE_DEGREES):
            return -1

        if cart_position < -0.8:
            discrete_state = 0
        elif cart_position < 0.8:
            discrete_state = 1
        else:
            discrete_state = 2

        if cart_velocity < 0.5:
            discrete_state += 3
        elif cart_velocity >= -0.5:
            discrete_state += 6

        if pole_angle < -SIX_DEGREES:
            pass
        elif pole_angle < -ONE_DEGREE:
            discrete_state += 9
        elif pole_angle < 0:
            discrete_state += 18
        elif pole_angle < ONE_DEGREE:
            discrete_state += 27
        elif pole_angle < SIX_DEGREES:
            discrete_state += 36
        else:
            discrete_state += 45

        if pole_velocity < -FIFTY_DEGREES:
            pass
        elif pole_velocity < FIFTY_DEGREES:
            discrete_state += 54
        else:
            discrete_state += 108

        return discrete_state

    def predict_action(self, state):
        discrete_state = self.discretize(state)

        if discrete_state == -1:
            action = -1
        elif random.uniform(0, 1) < EXPLORATION:
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

        #if self.epsilon > EXPLORATION_MIN:
        #    self.epsilon *= 0.9

# TODO: DQNAgent (on a different file)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = QLAgent(env.observation_space.shape[0], env.action_space.n)

    for i in range(100000):
        state = env.reset()
        #env.render()

        epochs = 0
        done = False

        while not done:
            action = agent.predict_action(state)

            if action == -1:
                agent.update_table(state, state, i%2==0, -10)
                break

            next_state, reward, done, info = env.step(action)

            agent.update_table(state, next_state, action, reward)

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print("Simulation {} ended in {} epochs".format(i, epochs))
