import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
from drl_agent import SimpleAgent, BatchAgent, RandomBatchAgent

# Numero de iteraciones que competiran las dos redes para ver cual es mejor
BESTOF = 10 

# Compara los resultados de agent1 con los de agent2 y devuelve la mejor
def competir(agent1,  agent2):
        r1 = 0
        r2 = 0
        for i in range(BESTOF):
            v1 = lanzar(agent1)
            v2 = lanzar(agent2)
            print("ronda", i, ":V1 ", v1, " vs V2 ", v2)
            r1 += v1
            r2 += v2

        if r1 > r2 :
            a = agent1
            print("Agente1 es mejor")
        else:
            a = agent2
            print("Agente2 es mejor")

        return a

# Simuala una partida sin "aprender"
def lanzar(ag):
    env = gym.make('CartPole-v1')
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]]) 
    done = False
    acc_reward = 0
    while not done:

        action = ag.predict_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO: check this

        #agent.update_table(state, action, reward, next_state, done)
        #agentApprentice.update_table(state, action, reward, next_state, done)
        state = next_state
        acc_reward += reward
        
    return acc_reward

EPISODES = 1000

# cada COMP partidas se resalizara una comparacion para ver que red funcion mejor
COMP = 100

def simulate(env, agent):
    scores = deque(maxlen=100)
    performance = []

    agentApprentice = copy.copy(agent)
    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        if i % COMP == 0 :
            # En la primera etapa aprenderá directamente la red de Agent, no tiene sentido compararla con AgenteApprentice
            if i > COMP : 
                agent = competir(agent, agentApprentice)

            agentApprentice = copy.copy(agent)
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

            # en las primeras COMP partidas la red de Agent aprendera de forma normal, 
            # pero a partir de la COMP+1 aprendera AgenteAprendiz y cada COMP partidas 
            # se compararan para seguir etrenando a la mejor de las dos
            if i > COMP :
                 agentApprentice.update_model(state, action, reward, next_state, done)
            else:
                agent.update_model(state, action, reward, next_state, done)
            

            state = next_state
            acc_reward += reward

        agent.update_hyperparameters()
        if agent.supports_replay():
            agent.replay()

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
        results[i] = list(map(lambda x: 200 if x > 200 else x, results[i]))  # Limit the performance to 200
        ax.plot(results[i], color=colors[i])

    return plt


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    results = []
    # Uncomment the agent that you want to test
    #results.append(simulate(env, SimpleAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, BatchAgent(env.observation_space.shape[0], env.action_space.n)))
    results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n)))

    # Plot the results
    plt = create_plot(results)
    plt.show()
