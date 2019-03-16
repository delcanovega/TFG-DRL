import gym
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from rl_agent import QLAgent


EPISODES = 500


def simulate(env, agent):
    scores = deque(maxlen=100)
    performance = []

    for i in range(EPISODES):
        state = env.reset()

        done = False
        acc_reward = 0
        while not done:
            # if i > 1500:
            #    env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.update_table(state, next_state, action, reward)

            state = next_state
            acc_reward += reward

        agent.decrease_exploration()

        scores.append(acc_reward)
        performance.append(np.mean(scores))
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

    return performance


def create_plot(results):
    colors = ['blue', 'purple', 'pink']

    _, ax = plt.subplots()
    ax.set(xlabel='Episodio', ylabel='Media de todos los episodios')
    ax.grid()

    ax.plot([195 for i in range(len(results[0]))], color='green')
    for i in range(len(results)):
        # results[i] = list(map(lambda x: 200 if x > 200 else x, results[i]))  # Limit the performance to 200
        ax.plot(results[i], color=colors[i])

    return plt


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    results = []
    # Uncomment the configuration that you want to test
    results.append(simulate(env, QLAgent((1, 1, 3, 6), env.observation_space, env.action_space.n)))
    results.append(simulate(env, QLAgent((3, 3, 3, 6), env.observation_space, env.action_space.n)))
    results.append(simulate(env, QLAgent((1, 1, 5, 8), env.observation_space, env.action_space.n)))

    # Plot the results
    plt = create_plot(results)
    plt.show()
