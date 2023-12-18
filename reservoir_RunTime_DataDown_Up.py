import os
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from reservoirpy.nodes import Reservoir, Ridge
from metrics import Metrics
import pickle

MAXLEN = 11202


def normalization(s, needStd=False):
    if needStd:
        return np.array((s-s.mean())/s.std())
    else:
        return np.array(s-s.mean())


def removeArts(signal):
    s = signal.copy()
    mean = np.mean(s)
    normS = normalization(s, True)
    for i in range(5200, 5350):
        if abs(normS[i]) > 0.7:
            s[i] = s[i-150]
    normS = normalization(s, True)
    for i in range(0, 5200):
        if abs(normS[i]) > 5:
            s[i] = mean
    for i in range(6000, len(s)):
        if abs(normS[i]) > 5:
            s[i] = mean
    normS = normalization(s, True)
    for i in range(0, 5200):
        if abs(normS[i]) > 5:
            s[i] = mean
    for i in range(6000, len(s)):
        if abs(normS[i]) > 5:
            s[i] = mean
    normS = normalization(s, True)
    for i in range(5200, 5350):
        if abs(normS[i]) > 0.5:
            s[i] = s[i-150]
    return s


def normalize(a):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(a)


def gausFilter(signal, sigma=35):
    min1 = np.min(signal)
    gf = gaussian_filter(signal, sigma)
    min2 = np.min(gf)
    return gf*min1/min2


def loadCA(records, postfix):
    ca = {}
    for k in records.keys():
        rec = []
        for sig in records[k]:
            for key in sig.keys():
                if str(key).endswith(postfix):
                    rec.append(sig[str(key)][:, 1])
        ca[k] = rec
    return ca


# good --- is a dictionary where key is name folder and value is a file
def loadSignalInDict(good):
    records = {}
    for amp in good.keys():
        rec = []
        for j in good[amp]:
            rec.append(sio.loadmat(j))
        records[amp] = rec

    # load in са1
    ca1 = loadCA(records, '_2')

    # load in са3
    ca3 = loadCA(records, '_3')
    return ca1, ca3


def normed(ca):
    norm = {}
    for i in ca.keys():
        wa = np.array([removeArts(rec) for rec in ca[i]])
        normca = np.array([normalization(rec) for rec in wa])
        norm[i] = normca
    return norm


def filter(norm):
    filt = {}
    for k in norm.keys():
        filt[k] = np.array([normalization(gausFilter(rec), True)
                           for rec in norm[k]])
    return filt


def average(filt):
    averaged = []
    for amp in filt.keys():
        tmp = np.array(filt[amp])
        averaged.append(sum(tmp)/len(tmp))  # ???
    return averaged


def createTrainModel(averaged, numTest):
    caTest = averaged[numTest].reshape(-1,1)
    
    Xtr = np.delete(averaged, numTest, 0)
    caTrain = []
    for i in range(len(Xtr)):
        caTrain.append(Xtr[i].reshape(-1,1))
    return caTest, caTrain


def reservoirRun(param, ca1Train, ca3Train, ca3Test, segment):
    reservoir = Reservoir(param[0], lr=param[1],
                          sr=param[2], activation='relu') ## relu ~ function Hevisaide f(x) = if x > 0: x else: 0
    readout = Ridge(ridge=param[3])
    #print(ca3Train)
    for i in range(len(ca3Train)):
        train_states = reservoir.run(
            ca3Train[i], reset=False)
        readout = readout.fit(train_states, ca1Train[i])

    test_states = reservoir.run(ca3Test)
    return readout.run(test_states)


def createWorkDirectory(runPath, NUMBERTEST):
    if runPath not in os.listdir(os.getcwd()):
        print(f"Path \"{runPath}\" not in current working directory")
        print(f"Path \"{runPath}\" created")
        os.mkdir(runPath)
    runPath += '/'

    if str(NUMBERTEST) in os.listdir(f'./{runPath}'):
        return runPath + str(NUMBERTEST) + '/'

    runPath += str(NUMBERTEST) + '/'
    os.mkdir(runPath)
    # os.mkdir(runPath+'ca1')
    # os.mkdir(runPath+'ca3')
    os.mkdir(runPath+'plot')
    return runPath


def plotSignal(true1, pred1, coeff1_1, coeff2_1, trueM1, predM1, numTest):
    plt.subplot().clear()
    plt.close()
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=400)
    fig.subplots_adjust(left=0.2)
    x1 = np.linspace(0, len(true1)/20, len(true1))
    y1 = np.linspace(0, len(pred1)/20, len(pred1))

    left = 0
    right = MAXLEN/20
    ax.set_ylim(np.min(np.hstack([true1, pred1])),
                np.max(np.hstack([true1, pred1])))
    ax.set_xlim([left, right])
    ax.plot(x1, true1, label="true")
    ax.plot(y1, pred1, label="pred")

    metr = 0.2*(coeff1_1[0] - coeff2_1[0])**2 + \
        0.4*(coeff1_1[1] - coeff2_1[1])**2 + \
        0.1*(coeff1_1[2] - coeff2_1[2])**2 + \
        0.3*(coeff1_1[3] - coeff2_1[3])**2

    ax.set_title(f'Test {int(numTest)}. \
metric = 0.2*({coeff1_1[0]:.2f} - {coeff2_1[0]:.2f})**2 + \
0.4*({coeff1_1[1]:.2f} - {coeff2_1[1]:.2f})**2 + 0.1*({coeff1_1[2]:.2f} - \
{coeff2_1[2]:.2f})**2 + 0.3*({coeff1_1[3]:.2f} - {coeff2_1[3]:.2f})**2 \
=  {metr}')

    ax.plot(np.linspace(trueM1[-2]/20, trueM1[-1]/20, (trueM1[-1]-trueM1[-2])),
            [true1[trueM1[-2]] for i in range(trueM1[-2], trueM1[-1])],
            label="true HW", color='green')

    ax.axvline(trueM1[1]/20, label="true TD", color='red')
    ax.axvline(trueM1[2]/20, label="true TR", color='purple')

    ax.plot(np.linspace(predM1[-2]/20, predM1[-1]/20, (predM1[-1]-predM1[-2])),
            [pred1[predM1[-2]] for i in range(predM1[-2], predM1[-1])],
            label="pred HW", color='green')

    ax.axvline(predM1[1]/20, label="pred TD", color='red')
    ax.axvline(predM1[2]/20, label="pred TR", color='blue')
    ax.set_ylabel("V, normalized")
    ax.set_xlabel("t, seconds")
    ax.legend()

    return fig, metr


def loadBestParam(NAME_TEST, best_param):
    if ("bestParam.pkl" in os.listdir(NAME_TEST)):
        with open(NAME_TEST + "/bestParam.pkl", 'rb') as f:
            best_param = pickle.load(f)

    return best_param


def saveBestParam(NAME_TEST, best_param):
    with open(NAME_TEST + "/bestParam.pkl", 'wb') as f:
        pickle.dump(best_param, f)


def runAlg(PATH_RUN, NAME_TEST, averaged1, averaged3, param, best_param, segment):
    runPath = createWorkDirectory(PATH_RUN, NAME_TEST)

    best_param = loadBestParam(runPath, best_param)

    for numTest, rate in zip(range(len(averaged1)), best_param):
        _, ca1Train = createTrainModel(averaged1, numTest)  # y
        ca3Test, ca3Train = createTrainModel(averaged3, numTest)  # x

        ca3Pred = reservoirRun(param, ca1Train, ca3Train, ca3Test, segment)

        # Переписать
        m = Metrics(MAXLEN)
        true1 = averaged1[numTest]
        
        ca3Pred = np.reshape(ca3Pred, len(ca3Pred))
        pred1 = ca3Pred

        trueM1 = m.printMetrics(true1)
        predM1 = m.printMetrics(pred1)

        coeff1_1 = m.getMetrics(pred1, averaged1, averaged3, 'ndarray')
        coeff2_1 = m.getMetrics(true1, averaged1, averaged3, 'ndarray')


        fig, metr = plotSignal(true1, pred1, coeff1_1,
                               coeff2_1, trueM1, predM1, numTest)

        if metr <= best_param[rate][1]:
            best_param[rate][0] = param
            best_param[rate][1] = metr
            fig.savefig(runPath+f'plot/{numTest}.pdf', bbox_inches='tight')

    print(best_param)
    saveBestParam(runPath, best_param)


if __name__ == '__main__':
    DATA_DOWN_PATH = 'DATA_DOWN'
    DATA_UP_PATH = 'DATA_UP'
    DOWN_TEST = 'DOWN_PAIRED'
    UP_TEST = 'UP_PAIRED'
    PATH_RUN = 'TEST'

    best_param = {200: [[], float(100)],
                  300: [[], float(100)], 400: [[], float(100)],
                  500: [[], float(100)], 1000: [[], float(100)]}

    # units: int = None, 0 < lr <= 1: float = 1, sr: float | None = None, ridge
    param = [65, 0.5, 0.01, 1e-7]
    start, stop = 0, 10000
    segment = [start, stop]

    averaged1 = np.load('./' + DATA_DOWN_PATH + '/CA1_DOWN_PAIRED.npy')
    averaged1 = np.delete(averaged1, 0, 0)

    averaged3 = np.load('./' + DATA_DOWN_PATH + '/CA3_DOWN_PAIRED.npy')
    averaged3 = np.delete(averaged3, 0, 0)

    
    runPath = createWorkDirectory(PATH_RUN, DOWN_TEST)
    runAlg(PATH_RUN, DOWN_TEST, averaged1, averaged3, param, best_param, segment)

    best_param = {200: [[], float(100)],
                  300: [[], float(100)], 400: [[], float(100)],
                  500: [[], float(100)], 1000: [[], float(100)]}

    averaged1 = np.load('./' + DATA_UP_PATH + '/CA1_UP_PAIRED.npy')
    averaged1 = np.delete(averaged1, 0, 0)

    averaged3 = np.load('./' + DATA_UP_PATH + '/CA3_UP_PAIRED.npy')
    averaged3 = np.delete(averaged3, 0, 0)

    runPath = createWorkDirectory(PATH_RUN, UP_TEST)
    runAlg(PATH_RUN, UP_TEST, averaged1, averaged3, param, best_param, segment)
