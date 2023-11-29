import numpy as np


class Metrics:
    def __init__(self, maxlen) -> None:
        self.maxlen = maxlen

    def getProperty(self, signal):
        self.left = self.right = self.argmin =\
            self.semiLeft = self.semiRight = np.argmin(signal)

        self.minValue = signal[self.argmin]
        self.semiMin = self.minValue/2
        mean = np.mean(signal)

        found = False
        while signal[self.left] < mean and self.left > 0:
            self.left -= 1
            if (signal[self.left] > self.semiMin) and not found:
                self.semiLeft = self.left
                found = True

        found = False
        while signal[self.right] < mean and self.right < len(signal) - 1:
            self.right += 1
            if (signal[self.right] > self.semiMin) and not found:
                self.semiRight = self.right
                found = True
        return 0

    def printMetrics(self, signal):
        self.getProperty(signal)
        SL = (self.argmin - self.semiLeft) / abs(self.minValue)
        return [SL, self.left, self.right, self.semiLeft, self.semiRight]

    def normMetrics(self, signal):  # критерий
        self.getProperty(signal)
        SL = (self.argmin - self.left) / self.minValue
        HW = self.semiRight - self.semiLeft
        TD = self.argmin - self.left
        TR = self.right - self.argmin
        return [SL, HW, TD, TR]

    def getMinMaxDictionary(self, filt1, filt3):
        SL = []
        HW = []
        TD = []
        TR = []
        for amp in filt1.keys():
            allMetrics = np.array([self.normMetrics(s) for s in filt1[amp]])
            for m in allMetrics:
                SL.append(m[0])
                HW.append(m[1])
                TD.append(m[2])
                TR.append(m[3])
        for amp in filt3.keys():
            allMetrics = np.array([self.normMetrics(s) for s in filt3[amp]])
            for m in allMetrics:
                SL.append(m[0])
                HW.append(m[1])
                TD.append(m[2])
                TR.append(m[3])
        minSL = min(SL)
        maxSL = max(SL)
        minHW = min(HW)
        maxHW = max(HW)
        minTD = min(TD)
        maxTD = max(TD)
        minTR = min(TR)
        maxTR = max(TR)
        return minSL, maxSL, minHW, maxHW, minTD, maxTD, minTR, maxTR

    def getMinMaxNDArray(self, filt1, filt3):
        SL = []
        HW = []
        TD = []
        TR = []
        allMetrics = np.array([self.normMetrics(
            filt1[s]) for s in range(len(filt1))])
        for m in allMetrics:
            SL.append(m[0])
            HW.append(m[1])
            TD.append(m[2])
            TR.append(m[3])

        allMetrics = np.array([self.normMetrics(filt3[s])
                              for s in range(len(filt3))])
        for m in allMetrics:
            SL.append(m[0])
            HW.append(m[1])
            TD.append(m[2])
            TR.append(m[3])
        minSL = min(SL)
        maxSL = max(SL)
        minHW = min(HW)
        maxHW = max(HW)
        minTD = min(TD)
        maxTD = max(TD)
        minTR = min(TR)
        maxTR = max(TR)
        return minSL, maxSL, minHW, maxHW, minTD, maxTD, minTR, maxTR

    def getMetrics(self, signal, filt1, filt3, strDict: str):
        if (strDict.lower() == "ndarray"):
            minSL, maxSL, minHW, maxHW, minTD, maxTD, minTR, maxTR \
                = self.getMinMaxNDArray(filt1, filt3)
        else:
            minSL, maxSL, minHW, maxHW, minTD, maxTD, minTR, maxTR \
                = self.getMinMaxDictionary(filt1, filt3)

        self.getProperty(signal)
        SL = (self.argmin - self.semiLeft) / abs(self.minValue)
        SL = (SL - minSL) / (maxSL - minSL)
        HW = self.semiRight - self.semiLeft
        HW = (HW - minHW) / (maxHW - minHW)
        TD = self.argmin - self.left
        TD = (TD - minTD) / (maxTD - minTD)
        TR = self.right - self.argmin
        TR = (TR - minTR) / (maxTR - minTR)
        return [SL, HW, TD, TR]
