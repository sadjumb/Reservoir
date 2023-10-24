import os
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from reservoirpy.nodes import Reservoir, Ridge
from metrics import Metrics

MAXLEN = 11202


def getInputData(path: str()):
    INPUT_DATA = [100, 200, 300, 400, 500, 1000]
    folders = os.listdir(path)
    good = {}
    for amp in INPUT_DATA:
        good[amp] = [path+folder+'/'+filename for folder in folders for filename in os.listdir(
            path=path+folder) if folder == f'{amp}']
    return good


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

        # if len(wa) > 0:
        # meanLine = sum(wa)/len(wa)              # ????
        # normMeanLine = sum(normca)/len(normca)  # ????
        # if i == 400:
        #    plt.plot(normca3[-1])
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


def createTrainModel(averaged, numTest, printShape=False):
    caTest = averaged[numTest]
    Xtr = np.delete(averaged, numTest, 0)
    caTrain = []
    for i in range(len(Xtr)):
        caTrain.append(Xtr[i])

    if printShape:
        print(f'caTest {np.shape(caTest)}')
        print(f'Xtr {np.shape(Xtr)}')
        print(f'caTrain {np.shape(caTrain)}')
        print()
    return caTest, caTrain


def reservoirRun(param, ridge, ca1Train, ca3Train, ca3Test, segment):
    reservoir = Reservoir(param[0], lr=param[1], sr=param[2])
    readout = Ridge(ridge=ridge)

    for i in range(len(ca3Train)):
        train_states = reservoir.run(
            ca3Train[i][segment[0]:segment[1]], reset=False)
        readout = readout.fit(train_states, ca1Train[i])

    test_states = reservoir.run(ca3Test[segment[0]:segment[1]])
    return readout.run(test_states)


def createWorkDirectory(NUMBERTEST):
    runPath = 'RUN'
    if runPath not in os.listdir(os.getcwd()):
        print(f"Path \"{runPath}\" not in current working directory")
        print(f"Path \"{runPath}\" created")
        os.mkdir(runPath)
    runPath += '/'

    if str(NUMBERTEST) in os.listdir(f'./{runPath}'):
        print(f"Path '{NUMBERTEST}' in ./{runPath} directory")
        os._exit(1)

    runPath += str(NUMBERTEST) + '/'
    os.mkdir(runPath)
    os.mkdir(runPath+'ca1')
    os.mkdir(runPath+'ca3')
    os.mkdir(runPath+'plot')
    return runPath


def paint(y1, y2, label1, label2, title):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(y1, label=label1, color="blue")
    ax.plot(y2, label=label2, color="red")
    ax.grid()
    fig.legend()
    ax.set_title(title)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    DATA_PATH = 'INPUT_DATA'
    NUMBERTEST = 0
    # numTest = 0
    param = [50, 0.55, 0.1]
    start, stop = 0, 10000
    segment = [start, stop]

    if DATA_PATH not in os.listdir(os.getcwd()):
        print(f"Path \"{DATA_PATH}\" not in current working directory")
        os._exit(1)
    DATA_PATH += '/'

    ca1, ca3 = loadSignalInDict(getInputData(DATA_PATH))
    runPath = createWorkDirectory(NUMBERTEST)

    normed1 = normed(ca1)
    normed3 = normed(ca3)

    filt1 = filter(normed1)
    filt3 = filter(normed3)

    averaged1 = average(filt1)
    averaged3 = average(filt3)

    for numTest in range(len(averaged1)):
        ca1Test, ca1Train = createTrainModel(averaged1, numTest)  # y
        ca3Test, ca3Train = createTrainModel(averaged3, numTest)  # x

        # units: int = None, 0 < lr <= 1: float = 1, sr: float | None = None,
        ca3Pred = reservoirRun(
            param, 1e-7, ca1Train, ca3Train, ca3Test, segment)

        # Переписать
        m = Metrics(MAXLEN)
        true1 = averaged1[numTest]
        pred1 = ca3Pred[0]

        trueM1 = m.printMetrics(true1)
        predM1 = m.printMetrics(pred1)

        coeff1_1 = m.getMetrics(pred1, filt1, filt3)
        coeff2_1 = m.getMetrics(true1, filt1, filt3)

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
        ax.set_title(f'Test {int(numTest)}. metric = 0.2*({coeff1_1[0]:.2f} - {coeff2_1[0]:.2f})**2 + 0.4*({coeff1_1[1]:.2f} - {coeff2_1[1]:.2f})**2 + 0.1*({coeff1_1[2]:.2f} - {coeff2_1[2]:.2f})**2 + 0.3*({coeff1_1[3]:.2f} - {coeff2_1[3]:.2f})**2 =  {0.2*(coeff1_1[0] - coeff2_1[0])**2 + 0.4*(coeff1_1[1] - coeff2_1[1])**2 + 0.1*(coeff1_1[2] - coeff2_1[2])**2 + 0.3*(coeff1_1[3] - coeff2_1[3])**2}')
        ax.plot(np.linspace(trueM1[-2]/20, trueM1[-1]/20, (trueM1[-1]-trueM1[-2])), [true1[trueM1[-2]]
                                                                                     for i in range(trueM1[-2], trueM1[-1])], label="true HW", color='green')
        ax.axvline(trueM1[1]/20, label="true TD", color='red')
        ax.axvline(trueM1[2]/20, label="true TR", color='purple')
        ax.plot(np.linspace(predM1[-2]/20, predM1[-1]/20, (predM1[-1]-predM1[-2])), [
                pred1[predM1[-2]] for i in range(predM1[-2], predM1[-1])], label="pred HW", color='green')
        ax.axvline(predM1[1]/20, label="pred TD", color='red')
        ax.axvline(predM1[2]/20, label="pred TR", color='blue')
        ax.set_ylabel("V, normalized")
        ax.set_xlabel("t, seconds")
        ax.legend()
        fig.savefig(runPath+f'plot/{numTest}.pdf', bbox_inches='tight')
        # plt.show()

    # paint(ca3Train[3], ca1Train[3], "CA3", "CA1", "Train CA1 and CA3")

    # runPath = createWorkDirectory(NUMBERTEST)
    #
