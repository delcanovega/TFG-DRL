import gym
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from drl_agent import SimpleAgent, RandomBatchAgent

EPISODES = 1000

def simulate(env, agent):
    scores = deque(maxlen=100)
    performance = []

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        done = False
        acc_reward = 0
        while not done:
            # if i > 1500:
            #    env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

            agent.update_model(state, action, reward, next_state, done)
            
            state = next_state
            acc_reward += reward

        agent.update_hyperparameters()

        scores.append(acc_reward)
        performance.append(np.mean(scores))
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

        if agent.supports_replay():
            agent.replay(100)
    
    return performance


def create_plot(results):
    colors = ['blue', 'purple', 'pink']

    _, ax = plt.subplots()
    ax.set(xlabel='Episodio', ylabel='Media de todos los episodios')
    ax.grid()

    ax.plot([195 for i in range(len(results[0]))], color='green')
    for i in range(len(results)):
        results[i] = list(map(lambda x: 200 if x > 200 else x, results))  # Limit the performance to 200
        ax.plot(results[i], color=colors[i])

    return plt


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    results_sa = simulate(env, SimpleAgent(env.observation_space.shape[0], env.action_space.n))
    # TODO:
    # results_ba = simulate(env, BatchAgent(env.observation_space.shape[0], env.action_space.n))
    results_rb = simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n))

    # Plot the results
    plt = create_plot([results_sa, results_rb])
    plt.show()
