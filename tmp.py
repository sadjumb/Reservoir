import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    DATA_PATH = 'DATA_DOWN'
    averaged1 = np.load('./' + DATA_PATH + '/CA1_DOWN_PAIRED.npy')
    averaged3 = np.load('./' + DATA_PATH + '/CA3_DOWN_PAIRED.npy')
    plt.plot(averaged1[0])
    plt.plot(averaged3[1])
    plt.show()
    print(averaged1)
