import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
import reservoirpy as rpy
import pandas as pd

MAXLEN = 11202


def getPaths(path: str()):
    INPUT_DATA = [100, 200, 300, 400, 500, 1000]
    folders = os.listdir(path)
    good = {}
    for amp in INPUT_DATA:
        good[amp] = [DATAPATH+folder+'/'+filename for folder in folders for filename in os.listdir(
            path=DATAPATH+folder) if folder == f'{amp}']
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

        if len(wa) > 0:
            meanLine1 = sum(wa)/len(wa)
            normMeanLine1 = sum(normca)/len(normca)
        # if i == 400:
        #    plt.plot(normca3[-1])
        norm[i] = normca
    return norm


if __name__ == '__main__':
    DATAPATH = 'INPUT_DATA'
    print(os.listdir(os.getcwd()))

    if DATAPATH not in os.listdir(os.getcwd()):
        print(f"Path \"{DATAPATH}\" not in current working directory")
        os._exit(1)

    DATAPATH += '/'
    ca1, ca3 = loadSignalInDict(getPaths(DATAPATH))

    normed1 = normed(ca1)
    normed3 = normed(ca3)
    # df = pd.DataFrame.from_dict(ca1)
    #
    # df.to_csv()
