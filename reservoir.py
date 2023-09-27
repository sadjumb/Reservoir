import os
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from reservoirpy.nodes import Reservoir, Ridge

MAXLEN = 11202


def getPaths(path: str()):
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


def createTrainModel(averaged, num_test):
    caTest = averaged[num_test]
    X_tr = np.delete(averaged, num_test, 0)
    caTrain = []
    for i in range(len(X_tr)):
        caTrain.append(X_tr[i])

    print(f'caTest {np.shape(caTest)}')
    print(f'X_tr {np.shape(X_tr)}')
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
    NUMBER_RUN = 0
    numTest = 0

    if DATA_PATH not in os.listdir(os.getcwd()):
        print(f"Path \"{DATA_PATH}\" not in current working directory")
        os._exit(1)
    DATA_PATH += '/'

    ca1, ca3 = loadSignalInDict(getPaths(DATA_PATH))

    normed1 = normed(ca1)
    normed3 = normed(ca3)

    filt1 = filter(normed1)
    filt3 = filter(normed3)

    averaged1 = average(filt1)
    averaged3 = average(filt3)
    # print(np.shape(averaged1))

    print('ca1 shape: ')
    ca1Test, ca1Train = createTrainModel(averaged1, numTest)  # y
    print('ca3 shape: ')
    ca3Test, ca3Train = createTrainModel(averaged3, numTest)  # x

    # paint(ca3Train[3], ca1Train[3], "CA3", "CA1", "Train CA1 and CA3")

    # runPath = createWorkDirectory(NUMBERTEST)
    # fig.savefig(runPath+'plot/TrainCA1_CA3.jpeg', bbox_inches='tight')

    # units: int = None, 0 < lr <= 1: float = 1, sr: float | None = None,
    param = [50, 0.55, 0.1]
    start, stop = 0, 10000
    segment = [start, stop]
    ca3Pred = reservoirRun(
        param, 1e-7, ca1Train, ca3Train, ca3Test, segment)

    # print(np.shape(Y_pred.reshape(len(Y_pred[0]), -1)))
    paint(ca3Pred[0], ca1Test, "Predicted CA1", "Real CA1", "Test")
