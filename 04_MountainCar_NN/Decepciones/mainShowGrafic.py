import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

from collections import deque


def create_mega_plot_fromfile():
    colors = ['blue', 'red', 'purple', 'green', 'pink']

    scores = np.fromfile("Scores.txt")
    real_scores = np.fromfile("Enviroment_Scores.txt")
    performance = np.fromfile("Performance.txt")
    maxVel = np.fromfile("MaxVel.txt")
    maxPos = np.fromfile("MaxPos.txt")
    velAcc = np.fromfile("VelAcc.txt")
    posAcc = np.fromfile("PosAcc.txt")
    
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
    
    plt = create_mega_plot_fromfile()
    plt.show()
