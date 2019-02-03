#coding=UTF-8

# Imports
import gym
# import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from NN_CartPole import DQNAgent

# Hyperparameters
LEARNING_RATE = 1.0
EXPLORATION = 1.0
DISCOUNT_FACTOR = 0.95

EPISODES = 500

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]]) # TODO: check this

        done = False
        acc_reward = 0

        while not done:
            # if i > 1500:
            #    env.render()

            action = agent.predict_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]]) # TODO: check this
            agent.update_table(state, action, reward, next_state, done)
            state = next_state
            acc_reward += reward

        print("Episode {}/{} ended with score {}".format(i+1, EPISODES, acc_reward))
        agent.replay(32)

        # if i % 100 == 0 and i > 0:
        #     print("Simulations {}-{} ended with {} average score".format(i - 100, i, round(acc_reward / 100)))
        #     acc_reward = 0

    # # Plot the results
    # performance = list(map(lambda x: 200 if x > 200 else x, performance))  # Limit the performance to 200
    #
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Episodio', ylabel='Media de los últimos 100 episodios')
    # ax.grid()
    #
    # ax.plot(performance, color='blue')
    # ax.plot([195 for i in range(len(performance))], color='green')
    #
    # plt.show()
