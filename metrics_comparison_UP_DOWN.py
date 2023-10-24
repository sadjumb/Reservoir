import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def loadBestParam(NAME_TEST, NAME_DICT):
    best_param = {200: [[], float(100)],
                  300: [[], float(100)], 400: [[], float(100)],
                  500: [[], float(100)], 1000: [[], float(100)]}

    if (NAME_DICT in os.listdir('./RUN/' + NAME_TEST)):
        with open('./RUN/' + NAME_TEST + "/bestParam.pkl", 'rb') as f:
            best_param = pickle.load(f)

    return best_param


if __name__ == "__main__":
    DOWN_TEST = 'DOWN_PAIRED'
    NAME_DOWN_DICT = "bestParam.pkl"
    down = loadBestParam(DOWN_TEST, NAME_DOWN_DICT)

    UP_TEST = 'UP_PAIRED'
    NAME_UP_DICT = "bestParam.pkl"
    up = loadBestParam(UP_TEST, NAME_UP_DICT)
    print(f'Up parametrs: {up}')
    print(f'Down parametrs{down}')

    metricUp = np.array([up[val][1] for val in up])
    metricDown = np.array([down[val][1] for val in down])
    xs = np.array([i for i in up.keys()])

    plt.figure(figsize=(5, 5), dpi=400)

    plt.plot(xs, metricUp)
    plt.scatter(xs, metricUp, s=50, marker='o', label='CA3 Up')
    plt.plot(xs, metricDown)
    plt.scatter(xs, metricDown, s=50, marker='o', label='CA3 Down')
    plt.legend()

    plt.ylabel("metric", fontsize=15)
    plt.xlabel("stimulation amplitude, μA", fontsize=15)
    plt.savefig("metrics.png", bbox_inches='tight')
