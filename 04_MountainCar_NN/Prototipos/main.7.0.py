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

def represent(arr, mt):
    return arr[mt]

t = np.arange(EPISODES)

def ourReward(state, max):
    # trampeamos la reward
    # Para que no penalice la direccion lo ponemos en valor absoluto
    reward = abs(state[0][0]+0.5) * 10

    # le damos un beneficio local que esperemos que le ayude (el agua del raton)
    #if abs(state[0][0]+0.5) >= 0 and max:
    #    reward += 10

    return reward


def simulate(env, agent, use_apprentice=False):
    
    scores = []
    real_scores = []
    maxVel = [] 
    maxPos = []
    stepsList = []
    velAcc = []
    posAcc = []
    
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
        acc_reward_environment = 0
        acc_vel = 0
        acc_pos = 0
        maxH = 0
        maxp = 0
        maxv = 0

        step = 0
        while not done:
            if i > 00:
                env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            step += 1
            acc_reward_environment += reward
            acc_pos += abs(state[0][0])
            acc_vel += abs(state[0][1])
            #probamos a no darle la bonificacion
            if abs(state[0][0]) > maxH:
                max = True
                maxH = abs(state[0][0])
            else:
                max = False

            if maxp < state[0][0]:
                maxp = state[0][0]

            if maxv < abs(state[0][1]) :
                maxv = state[0][1]

            # (reward ==0 => win )
            if done and step < 200:
                acc_reward = 1000 - step
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
        real_scores.append(acc_reward_environment)
        maxVel.append(maxv)
        maxPos.append(maxp)
        stepsList.append(step)
        velAcc.append(acc_vel)
        posAcc.append(acc_pos)
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

    return scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc


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

def create_mega_plot(scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc):
    colors = ['blue', 'red', 'purple', 'green', 'pink']

    ax = plt.subplot(311)
    ax.set(xlabel='Episodio')
    ax.plot(range(len(scores)),scores, color=colors[0])
    ax.plot(range(len(scores)),real_scores, color=colors[1])
    #ax.plot(range(len(scores)),stepsList, color=colors[4])

    a2x = plt.subplot(323)
    a2x.set(xlabel='Episodio')
    a2x.plot(range(len(scores)),maxVel, color=colors[2])
    
    a3x = plt.subplot(324)
    a3x.set(xlabel='Episodio')
    a3x.plot(range(len(scores)),maxPos, color=colors[3])
    
    a4x = plt.subplot(325)
    a4x.set(xlabel='Episodio')
    a4x.plot(range(len(scores)),velAcc, color=colors[2])
    
    a5x = plt.subplot(326)
    a5x.set(xlabel='Episodio')
    a5x.plot(range(len(scores)),posAcc, color=colors[3])
    
    #ax.plot([195 for i in range(len(results[0]))], color='green')
    #for i in range(len(scores)):
        # results[i] = list(map(lambda x: 200 if x > 200 else x, results[i]))  # Limit the performance to 200
    #    ax.plot(scores[i], color=colors[0])
    #    ax.plot(real_scores[i], color=colors[1])
    #    ax.plot(maxVel[i], color=colors[2])
    #    ax.plot(maxPos[i], color=colors[3])
    #    ax.plot(stepsList[i], color=colors[4])#score
        

    return plt



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    #results = []
    # Uncomment the agent that you want to test
    #results.append(simulate(env, SimpleAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, BatchAgent(env.observation_space.shape[0], env.action_space.n)))
    #results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n)))
    # results.append(simulate(env, RandomBatchAgent(env.observation_space.shape[0], env.action_space.n), True))
    #results.append(simulate(env, RandomBatchAgentTwoBrains(env.observation_space.shape[0], env.action_space.n)))
    scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc = simulate(env,
     RandomBatchAgentTwoBrains(env.observation_space.shape[0], env.action_space.n))



    # Plot the results
    #plt = create_plot(results)
    plt = create_mega_plot(scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc)
    plt.show()
