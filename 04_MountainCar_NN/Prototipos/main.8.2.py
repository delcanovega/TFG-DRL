import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server
from drl_agent_6 import RandomBatchAgentTwoBrainsBestSave


EPISODES = 1500

COMP = 10    # Every COMP episodes the agent and apprentice are compared
BESTOF = 20  # Number of simulations of the comparison
# Each episode the reward recived will be reduce to reforce the fastest way to reach our goal
PENATYFORSLOW = 0.999


# basada en la velocidad absoluta y da una bonificacion si alcanza una distancia mayor de lo que ha alcanzado hasta ahora

def represent(arr, mt):
    return arr[mt]

t = np.arange(EPISODES)

def ourReward(state):
    reward = abs(state[0][1]) * 100
    return reward

def subSimulate(env, agent):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    apprenticeReward = 0
    
    while not done:
        action = agent.optimal_action(state, 1)

        next_state, reward, done, _ = env.step(action)
        
        next_state = np.reshape(
            next_state, [1, env.observation_space.shape[0]])

        reward = ourReward(state)
        state = next_state
        apprenticeReward += reward

    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    bestReward = 0
    while not done:
        action = agent.optimal_action(state, 2)

        next_state, reward, done, _ = env.step(action)
        
        next_state = np.reshape(
            next_state, [1, env.observation_space.shape[0]])
        
        reward = ourReward(state)
        state = next_state
        bestReward += reward

    return apprenticeReward, bestReward

def simulate(env, agent, use_apprentice=False):
    oldScores = deque(maxlen=100) 
    performance, scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, competes, bestchangedPos, bestchangedVal, apprenticeScores, bestScores = [[] for _ in range(13)]
    
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
        apprenticeReward, bestReward = subSimulate(env, agent)
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
        apprenticeScores.append(apprenticeReward)
        bestScores.append(bestReward)
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))
    
    acc_mentor, acc_apprentice, acc_best = agent.finalTrial(env)

    return scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores


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

def saveScores(scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores):
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
    np.savetxt('apprenticeScores.txt',apprenticeScores)
    np.savetxt('bestScores.txt',bestScores)
    np.savetxt('acc_mentor.txt',acc_mentor)
    np.savetxt('acc_apprentice.txt',acc_apprentice)
    np.savetxt('acc_best.txt',acc_best)

def loadScores():
    
    scores = np.loadtxt('Scores.txt')
    real_scores = np.loadtxt("Enviroment_Scores.txt")
    maxVel = np.loadtxt("MaxVel.txt")
    maxPos = np.loadtxt("MaxPos.txt")
    velAcc = np.loadtxt("VelAcc.txt")
    posAcc = np.loadtxt("PosAcc.txt")
    performance = np.loadtxt("Performance.txt")
    competes = np.loadtxt("Competes.txt")
    bestchangedPos = np.loadtxt("Best_changed_Pos.txt")
    bestchangedVal = np.loadtxt("Best_changed_Val.txt")
    apprenticeScores = np.loadtxt('apprenticeScores.txt')
    bestScores = np.loadtxt('bestScores.txt')
    acc_mentor = np.loadtxt('acc_mentor.txt')
    acc_apprentice = np.loadtxt('acc_apprentice.txt')
    acc_best = np.loadtxt('acc_best.txt')

    return scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores

def create_mega_plot(scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores):
    colors = ['blue', 'red', 'purple', 'green', 'pink']

    ax = plt.subplot(211)
    ax.set(xlabel='Episodes')
    ax.plot([800 for i in range(len(scores))], 'm--', label = 'Goal')
    ax.plot(range(len(scores)),apprenticeScores, color = 'cyan', label = 'Apprentice Scores')
    ax.plot(range(len(scores)),bestScores, color = colors[3], label = 'Best Scores')
    ax.plot(range(len(scores)),scores, color=colors[0], label = 'Scores')
    ax.plot(range(len(real_scores)),real_scores, color=colors[1], label = 'Enviroment Scores')
    ax.plot(range(len(scores)), performance, color = 'black', label = '100 mean scores')
    ax.legend()
    #ax.legend(loc='lower right')
    

    #a2x = plt.subplot(223)
    #a2x.set(xlabel='Episodes')
    #a2x.plot(range(len(maxVel)),maxVel, color=colors[2], label = 'Maximum velocity')
    #a2x.legend()
    

    a3x = plt.subplot(223)
    a3x.set(xlabel='Episodes')
    a3x.plot(range(len(maxPos)),maxPos, color=colors[3], label = 'Maximum position')
    a3x.legend()

    #a4x = plt.subplot(325)
    #a4x.set(xlabel='Competitions')
    #a4x.set(ylabel='0 = mentor, 1 = apprendice')
    #a4x.plot(range(len(competes)), competes, 'kD', label = 'Winner of the competicion')   
    #a4x.plot(bestchangedPos, bestchangedVal, 'g^', label = 'Greater than the Best')
    #a4x.legend(loc = 'lower left')

    a5x = plt.subplot(224)
    a5x.set(xlabel='FINAL TRIAL')
    a5x.plot(range(len(acc_mentor)),acc_mentor, color=colors[0], label = 'FINAL MENTOR')
    a5x.plot(range(len(acc_apprentice)),acc_apprentice, color=colors[1], label = 'FINAL APPRENTICE')
    a5x.plot(range(len(acc_best)),acc_best, color=colors[3], label = 'FINAL BEST')
    a5x.legend()
    return plt

if __name__ == '__main__':
    #env = gym.make('MountainCar-v0')

    #scores, real_scores, maxVel, maxPos, stepsList, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores = simulate(env,
     #RandomBatchAgentTwoBrainsBestSave(env.observation_space.shape[0], env.action_space.n))
    #saveScores(scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores)


    # Plot the results
    #plt = create_plot(results)
    
    scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores = loadScores()
    plt = create_mega_plot(scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores)
    
    plt.show()
