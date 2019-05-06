import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
from drl_agent_2 import RandomBatchAgentTwoBrains


EPISODES = 1500

COMP = 10    # Every COMP episodes the agent and apprentice are compared
BESTOF = 20  # Number of simulations of the comparison
# Each episode the reward recived will be reduce to reforce the fastest way to reach our goal
PENATYFORSLOW = 0.999


# basada en la velocidad absoluta y da una bonificacion si alcanza una distancia mayor de lo que ha alcanzado hasta ahora


def ourReward(state, max):
    # trampeamos la reward
    # Para que no penalice la direccion lo ponemos en valor absoluto
    reward = state[0][1] * 100

    # le damos un beneficio local que esperemos que le ayude (el agua del raton)
    #if abs(state[0][0]+0.5) >= 0 and max:
    #    reward += 10

    return reward


def simulate(env, agent, use_apprentice=False):
    scores = deque(maxlen=100)
    performance = []
    #agent.load('modelos/MontainCar/modelo.h5')
    
    #this is realy important because almost all metods(4) has been changed in RandomBatchAgentTwoBrains
    use_apprentice = True if (isinstance(agent, RandomBatchAgentTwoBrains)) else False 
    
    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        # Do not compete on the first episode
        if use_apprentice and i % COMP == 0 and i >= COMP:
            agent.compete(env)
            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])

        done = False
        acc_reward = 0
        maxH = 0
        step = 0
        while not done:
            if i > 00:
                env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            step += 1

            #probamos a no darle la bonificacion
            if abs(state[0][0]) > maxH:
                max = True
                maxH = abs(state[0][0])
            else:
                max = False

            # (reward ==0 => win ) => reward final  = 900 - time to reach the goal
            if done and step < 200:
                acc_reward = 4000 - step
            else:
                reward = ourReward(state, max)

            #print(state[0], ";\t Recompensa EP", reward, ";\t  Acumulada ", acc_reward, ";\t Step ", step)

            next_state = np.reshape(
                next_state, [1, env.observation_space.shape[0]])

            # now Mentor and Apprentice share memory
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
    #results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n)))
    # results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n), True))
    results.append(simulate(env, RandomBatchAgentTwoBrains(env.observation_space.shape[0], env.action_space.n)))

    # Plot the results
    plt = create_plot(results)
    plt.show()
