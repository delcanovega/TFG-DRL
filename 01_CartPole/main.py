import gym
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from rl_agent import QLAgent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = QLAgent(env.observation_space, env.action_space.n)

    scores = deque(maxlen=100)
    performance = []
    for i in range(500):
        state = env.reset()

        done = False
        acc_reward = 0
        while not done:
            # if i > 1500:
            #    env.render()

            action = agent.predict_action(state)

            next_state, reward, done, info = env.step(action)
            agent.update_table(state, next_state, action, reward)

            state = next_state
            acc_reward += reward

        agent.decrease_exploration()
        scores.append(acc_reward)
        performance.append(np.mean(scores))

        # if i % 100 == 0 and i > 0:
        #     print("Simulations {}-{} ended with {} average score".format(i - 100, i, round(acc_reward / 100)))
        #     acc_reward = 0

    # Plot the results
    performance = list(map(lambda x: 200 if x > 200 else x, performance))  # Limit the performance to 200

    fig, ax = plt.subplots()
    ax.set(xlabel='Episodio', ylabel='Media de los últimos 100 episodios')
    ax.grid()

    ax.plot(performance, color='blue')
    ax.plot([195 for i in range(len(performance))], color='green')

    plt.show()
