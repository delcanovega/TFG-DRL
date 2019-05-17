import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
from drl_agent_4 import RandomBatchAgentTwoBrainsBestSave


EPISODES = 500

COMP = 10    # Every COMP episodes the agent and apprentice are compared
BESTOF = 20  # Number of simulations of the comparison
# Each episode the reward recived will be reduce to reforce the fastest way to reach our goal
PENATYFORSLOW = 0.999


# basada en la velocidad absoluta y da una bonificacion si alcanza una distancia mayor de lo que ha alcanzado hasta ahora

def represent(arr, mt):
    return arr[mt]

t = np.arange(EPISODES)

def ourReward(state):
    reward = abs(state[0][0]+0.5) * 10
    return reward


def simulate(env, agent, use_apprentice=False):
    oldScores = deque(maxlen=100) 
    performance, scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, competes, bestchangedPos, bestchangedVal = [[] for _ in range(11)]
    
    #agent.load('modelos/MontainCar/modelo.h5')
    
    use_apprentice = True if (isinstance(agent, RandomBatchAgentTwoBrainsBestSave)) else False 

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        # Do not compete on the first episode
        if use_apprentice and i % COMP == 0:
            if i % 100 == 0:
                a,b = agent.bigCompete(env)
                bestchangedPos.append(i//COMP)
                bestchangedVal.append(b)
            else :
                a = agent.compete(env)

            competes.append(a)
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
           
            if abs(state[0][0]) > maxH:
                maxH = abs(state[0][0])
          
            if maxp < state[0][0]:
                maxp = state[0][0]

            if maxv < abs(state[0][1]) :
                maxv = state[0][1]

            if done and step < 200:
                acc_reward = 1000 - step
            else:
                reward = ourReward(state)

            
            next_state = np.reshape(
                next_state, [1, env.observation_space.shape[0]])

            agent.update_model(state, action, reward, next_state, done)

            if use_apprentice and i < COMP:
                agent.replay_Mentor()
            elif agent.supports_replay():
                agent.replay()

            state = next_state
            acc_reward += reward

        agent.update_hyperparameters()
        
        oldScores.append(acc_reward)
        scores.append(acc_reward)
        real_scores.append(acc_reward_environment)
        maxVel.append(maxv)
        maxPos.append(maxp)
        stepsList.append(step)
        velAcc.append(acc_vel)
        posAcc.append(acc_pos)
        performance.append(np.mean(oldScores))
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))

    return scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal


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

def create_mega_plot(scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal):
    colors = ['blue', 'red', 'purple', 'green', 'pink']

    
    np.savetxt('Scores.txt',scores)
    np.savetxt('Enviroment_Scores.txt', real_scores)
    np.savetxt('MaxVel.txt',maxVel)
    np.savetxt('MaxPos.txt',maxPos)
    np.savetxt('VelAcc.txt',velAcc)
    np.savetxt('PosAcc.txt',posAcc)
    np.savetxt('Performance.txt',performance)
    np.savetxt('Competes.txt',competes)
    np.savetxt('Best_changed_Pos.txt',bestchangedPos)
    np.savetxt('Best_changed_Val.txt',bestchangedVal)
    

    ax = plt.subplot(311)
    ax.set(xlabel='Episodes')
    ax.plot([-200 for i in range(len(scores))], color=colors[4])
    ax.plot(range(len(scores)),scores, color=colors[0], label = 'Scores')
    ax.plot(range(len(scores)),real_scores, color=colors[1], label = 'Enviroment Scores')
    ax.plot(range(len(scores)), performance, color = 'black', label = '100 mean scores')
    ax.legend(loc='lower right')
    

    a2x = plt.subplot(323)
    a2x.set(xlabel='Episodes')
    #a2x.plot(range(len(scores)),velAcc, color='yellow', label = 'velocidad acumulada')
    a2x.plot(range(len(scores)),maxVel, color=colors[2], label = 'Maximum velocity')
    a2x.legend()
    

    a3x = plt.subplot(324)
    a3x.set(xlabel='Episodes')
    a3x.plot(range(len(scores)),maxPos, color=colors[3], label = 'Maximum position')
    a3x.legend()

    a4x = plt.subplot(325)
    a4x.set(xlabel='Competitions')
    a4x.set(ylabel='0 = mentor, 1 = apprendice')
    a4x.plot(range(len(competes)), competes, 'kD', label = 'Winner of the competicion')   
    a4x.plot(bestchangedPos, bestchangedVal, 'g^', label = 'Greater than the Best')
    a4x.legend()

    a5x = plt.subplot(326)
    a5x.set(xlabel='Episodes')
    a5x.plot(range(len(scores)),posAcc, color=colors[3], label = 'position accumulated')
    
    #ax.plot([195 for i in range(len(results[0]))], color='green')
    #for i in range(len(scores)):
        # results[i] = list(map(lambda x: 200 if x > 200 else x, results[i]))  # Limit the performance to 200
    #    ax.plot(scores[i], color=colors[0])
    #    ax.plot(real_scores[i], color=colors[1])
    #    ax.plot(maxVel[i], color=colors[2])
    #    ax.plot(maxPos[i], color=colors[3])
    #    ax.plot(stepsList[i], color=colors[4])#score
        

    return plt

def create_mega_plot_fromfile():
    colors = ['blue', 'red', 'purple', 'green', 'pink']

    scores = np.fromfile("Scores.txt")
    real_scores = np.fromfile("Enviroment_Scores.txt")
    maxVel = np.fromfile("MaxVel.txt")
    maxPos = np.fromfile("MaxPos.txt")
    velAcc = np.fromfile("VelAcc.txt")
    posAcc = np.fromfile("PosAcc.txt")
    performance = np.fromfile("Performance.txt")
    competes = np.fromfile("Competes.txt")
    bestchangedPos = np.fromfile("Best_changed_Pos.txt")
    bestchangedVal = np.fromfile("Best_changed_Val.txt")

    ax = plt.subplot(311)
    ax.set(xlabel='Episodes')
    ax.plot([-200 for i in range(len(scores))], color=colors[4])
    ax.plot(range(len(scores)),scores, color=colors[0], label = 'Scores')
    ax.plot(range(len(scores)),real_scores, color=colors[1], label = 'Enviroment Scores')
    ax.plot(range(len(scores)), performance, color = 'black', label = '100 mean scores')
    ax.legend()
    

    a2x = plt.subplot(323)
    a2x.set(xlabel='Episodes')
    #a2x.plot(range(len(scores)),velAcc, color='yellow', label = 'velocidad acumulada')
    a2x.plot(range(len(scores)),maxVel, color=colors[2], label = 'Maximum velocity')
    a2x.legend()
    

    a3x = plt.subplot(324)
    a3x.set(xlabel='Episodes')
    a3x.plot(range(len(scores)),maxPos, color=colors[3], label = 'Maximum position')
    a3x.legend()

    a4x = plt.subplot(325)
    a4x.set(xlabel='Competitions')
    a4x.set(ylabel='0 = mentor, 1 = apprendice')
    a4x.plot(range(len(competes)), competes, 'kD', label = 'Winner of the competicion')   
    a4x.plot(bestchangedPos, bestchangedVal, 'g^', label = 'Greater than the Best')
    a4x.legend()

    a5x = plt.subplot(326)
    a5x.set(xlabel='Episodes')
    a5x.plot(range(len(scores)),posAcc, color=colors[3], label = 'position accumulated')
    
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

    scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal = simulate(env,
     RandomBatchAgentTwoBrainsBestSave(env.observation_space.shape[0], env.action_space.n))



    # Plot the results
    #plt = create_plot(results)
    plt = create_mega_plot(scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal)
    plt.show()
