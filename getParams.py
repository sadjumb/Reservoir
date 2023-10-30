from reservoirpy.nodes import Reservoir, Ridge
import pickle
import numpy as np
import networkx as nx
import scipy.spatial
import matplotlib.pyplot as plt


def loadParam(PATH):
    with open(PATH + '/bestParam.pkl', 'rb') as f:
        return pickle.load(f)


def loadW(NAME_FILE):
    with open(NAME_FILE, 'rb') as f:
        return np.load(f)


if __name__ == '__main__':
    PATH_RUN = 'RUN_INT/'
    DATA_DOWN_PATH = 'DOWN_PAIRED/'
    DATA_UP_PATH = 'UP_PAIRED/'

    bestParamDown = loadParam(PATH_RUN+DATA_DOWN_PATH)
    bestParamUp = loadParam(PATH_RUN+DATA_UP_PATH)

    for ampl in bestParamUp.keys():
        print(f"Amplitude: {ampl}")
        print(f"Metrics: up metrics: {bestParamUp[ampl][1]}\t down metrics: {bestParamDown[ampl][1]}")
        print(f"Shape: up: {bestParamUp[ampl][0][0]}\t down: {bestParamDown[ampl][0][0]}")
        wDown = loadW(PATH_RUN+DATA_DOWN_PATH+f'W/{str(ampl)}.npy')
        print(f"wDown percent of the sparcity: {1.0 - (np.count_nonzero(wDown) / wDown.size)}")
        
        wUp = loadW(PATH_RUN+DATA_UP_PATH+f'W/{str(ampl)}.npy')
        print(f"wUp percent of the sparcity: {1.0 - (np.count_nonzero(wUp) / wUp.size)}")
        print()
        #plt.boxplot(scipy.spatial.distance_matrix(wDown, wUp, p=2))
        #plt.show()
        plt.close()


    #with open('RUN_INT/DOWN_PAIRED/W/200.npy', 'rb') as f:
    #    w = np.load(f)
    #print(bestParamDown)

    #matrW = 
    #sparcity = 1.0 - (np.count_nonzero(matrW) / matrW.size)
    #print(sparcity) # for Up and Down weights
