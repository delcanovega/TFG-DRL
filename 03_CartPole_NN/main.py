import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
from drl_agent import SimpleAgent, BatchAgent, RandomBatchAgent


EPISODES = 20

COMP = 10    # Every COMP episodes the agent and apprentice are compared
BESTOF = 20  # Number of simulations of the comparison

def compete(agent, apprentice, env):
    """Compares the agent and the apprentice returning the one that performs better"""

    total_reward_agent = 0
    total_reward_apprentice = 0
    for i in range(BESTOF):
        reward_agent = test(agent, env)
        reward_apprentice = test(apprentice, env)

        # print("Round {}: agent {} - apprentice {}".format(i + 1, reward_agent, reward_apprentice))
        total_reward_agent += reward_agent
        total_reward_apprentice += reward_apprentice

    print("Final results: agent {} - apprentice {}".format(total_reward_agent, total_reward_apprentice))
    return apprentice if total_reward_agent < total_reward_apprentice else agent 


def test(agent, env):
    """Simulates an episode with no learning"""

    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]]) 

    done = False
    acc_reward = 0
    while not done:
        action = agent.predict_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

        state = next_state
        acc_reward += reward
        
    return acc_reward


def simulate(env, agent, use_apprentice=False):
    scores = deque(maxlen=100)
    performance = []
    agent.load('modelos/CartPole/modelo.h5')

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        if use_apprentice and i % COMP == 0:
            # Do not compete on the first episode
            if i > COMP:
                agent = compete(agent, apprentice_agent, env)
                state = env.reset()
                state = np.reshape(state, [1, env.observation_space.shape[0]])
            apprentice_agent = copy.copy(agent)

        done = False
        acc_reward = 0
        while not done:
            # if i > 1500:
            #    env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

            if use_apprentice and i > COMP:
                apprentice_agent.update_model(state, action, reward, next_state, done)
                if apprentice_agent.supports_replay():
                    apprentice_agent.replay()
            else:
                agent.update_model(state, action, reward, next_state, done)
                if agent.supports_replay():
                    agent.replay()

            state = next_state
            acc_reward += reward

        agent.update_hyperparameters()

        scores.append(acc_reward)
        performance.append(np.mean(scores))
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

    agent.save('modelos/CartPole/modelo.h5')

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
    # Uncomment the agent that you want to test
    #results.append(simulate(env, SimpleAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, BatchAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n)))
    results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n), True))

    print(results)
    print('saving results...')
    np.savetxt("cartpole_results.txt", results, delimiter='\n', fmt='% 4d')
    # Plot the results
    plt = create_plot(results)
    plt.show()