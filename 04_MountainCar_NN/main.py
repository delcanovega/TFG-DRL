import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque

# Comment this to launch on server

from drl_agent import RandomBatchAgentTwoBrainsBestSave #using Adelta as optimizer


EPISODES = 10

COMP = 10    # Every COMP episodes the agent and apprentice are compared

# Each episode the reward recived will be reduce to reforce the fastest way to reach our goal
PENATYFORSLOW = 0.999


# basada en la VELOCIDAD ABSOLUTA * 100

def represent(arr, mt):
    return arr[mt]

t = np.arange(EPISODES)

def ourReward(state):
    reward = abs(state[0][1]) * 100
    return reward

def subSimulate(env, agent, k):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    acc_Reward = 0
    steps = 0
    while not done:
        action = agent.optimal_action(state, k)

        next_state, reward, done, _ = env.step(action)
        steps += 1
        next_state = np.reshape(
            next_state, [1, env.observation_space.shape[0]])

        if done and steps < 200:
            acc_Reward = 1000 - steps
        else:
            reward = ourReward(state)

        state = next_state
        acc_Reward += reward

    return acc_Reward

def simulate(env, agent, use_apprentice=False):
    oldScores = deque(maxlen=100) 
    performance, scores, real_scores, maxPos, apprenticeScores, bestScores = [[] for _ in range(6)]
    
    use_apprentice = True if (isinstance(agent, RandomBatchAgentTwoBrainsBestSave)) else False 

    for i in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        # Do not compete on the first episode
        if use_apprentice and i % COMP == 0:
            if i % 100 == 0:
                agent.bigCompete(env)
            else :
                agent.compete(env)

            state = env.reset()
            state = np.reshape(state, [1, env.observation_space.shape[0]])

        done = False
        acc_reward = 0
        acc_reward_environment = 0
        maxp = 0

        step = 0
        while not done:
            if i > 00:
                env.render()

            action = agent.predict_action(state)

            next_state, reward, done, _ = env.step(action)
            step += 1
            acc_reward_environment += reward
          
            if maxp < state[0][0]:
                maxp = state[0][0]

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
        apprenticeReward = subSimulate(env, agent, 1)
        bestReward = subSimulate(env, agent, 2)
        agent.update_hyperparameters()
        
        oldScores.append(acc_reward)
        scores.append(acc_reward)
        real_scores.append(acc_reward_environment)
        maxPos.append(maxp)
        performance.append(np.mean(oldScores))
        apprenticeScores.append(apprenticeReward)
        bestScores.append(bestReward)
        print("Episode {}/{} score {}".format(i + 1, EPISODES, int(acc_reward)))
    
    acc_mentor, acc_apprentice, acc_best = agent.finalTrial(env)

    return scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores


def saveScores(scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores):
    np.savetxt('Scores.txt',scores)
    np.savetxt('Enviroment_Scores.txt', real_scores)
    np.savetxt('MaxPos.txt',maxPos)
    np.savetxt('Performance.txt',performance)
    np.savetxt('apprenticeScores.txt',apprenticeScores)
    np.savetxt('bestScores.txt',bestScores)
    np.savetxt('acc_mentor.txt',acc_mentor)
    np.savetxt('acc_apprentice.txt',acc_apprentice)
    np.savetxt('acc_best.txt',acc_best)

def loadScores():
    
    scores = np.loadtxt('Scores.txt')
    real_scores = np.loadtxt("Enviroment_Scores.txt")
    maxPos = np.loadtxt("MaxPos.txt")
    performance = np.loadtxt("Performance.txt")
    apprenticeScores = np.loadtxt('apprenticeScores.txt')
    bestScores = np.loadtxt('bestScores.txt')
    acc_mentor = np.loadtxt('acc_mentor.txt')
    acc_apprentice = np.loadtxt('acc_apprentice.txt')
    acc_best = np.loadtxt('acc_best.txt')

    return scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores

def create_mega_plot(scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores):
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

    a3x = plt.subplot(223)
    a3x.set(xlabel='Episodes')
    a3x.plot(range(len(maxPos)),maxPos, color=colors[3], label = 'Maximum position')
    a3x.legend()

    a5x = plt.subplot(224)
    a5x.set(xlabel='FINAL TRIAL')
    a5x.plot(range(len(acc_mentor)),acc_mentor, color=colors[0], label = 'FINAL MENTOR')
    a5x.plot(range(len(acc_apprentice)),acc_apprentice, color=colors[1], label = 'FINAL APPRENTICE')
    a5x.plot(range(len(acc_best)),acc_best, color=colors[3], label = 'FINAL BEST')
    a5x.legend()
    return plt

if __name__ == '__main__':
    
    env = gym.make('MountainCar-v0')
    
    scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores = simulate(env,
     RandomBatchAgentTwoBrainsBestSave(env.observation_space.shape[0], env.action_space.n))

    #save scores
    #saveScores(scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores)

    #load scores
    #scores, real_scores, maxVel, maxPos, velAcc, posAcc, performance, competes,  bestchangedPos, bestchangedVal, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores = loadScores()
    
    plt = create_mega_plot(scores, real_scores, maxPos, performance, acc_mentor, acc_apprentice, acc_best, apprenticeScores, bestScores)
    plt.show()
