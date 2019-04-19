import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
import drl_agent
from drl_agent_1 import RandomBatchAgent
from drl_agent import RandomBatchAgentTwoBrains

EPISODES = 100

COMP = 10    # Every COMP episodes the agent and apprentice are compared
BESTOF = 20  # Number of simulations of the comparison

def simulate(env, agent):
    scores = deque(maxlen=100)
    performance = []
    agent.load('modelos/MontainCar/modelo.h5')

    #this is realy important because almost all metods(4) has been changed in RandomBatchAgentTwoBrains
    use_apprentice = True if (isinstance(agent, RandomBatchAgentTwoBrains)) else False 

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        # Do not compete on the first episode
        if use_apprentice and i % COMP == 0 and i >= COMP:
            agent.compete(env)
            
        done = False
        acc_reward = 0
        while not done:
            if i > 00:
               env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            
            #now Mentor and Apprentice share memory
            agent.update_model(state, action, reward, next_state, done)
           
            if use_apprentice and i < COMP:
                agent.replay_Mentor()
            elif agent.supports_replay():
                agent.replay()

            state = next_state
            acc_reward += reward

        agent.update_hyperparameters()

        scores.append(acc_reward)
        performance.append(np.mean(scores))
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

    agent.save('modelos/MontainCar/modelo.h5')

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
    env = gym.make('MountainCar-v0')

    results = []
    # Uncomment the agent that you want to test
    #results.append(simulate(env, SimpleAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, BatchAgent(env.observation_space.shape[0], env.action_space.n)))
    results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n), True))
    #results.append(simulate(env, RandomBatchAgentTwoBrains(env.observation_space.shape[0], env.action_space.n)))


    # Plot the results
    plt = create_plot(results)
    plt.show()