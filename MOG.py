import math
import random
import numpy as np
import math
from matplotlib import pyplot as plt

#计算高斯概率分布函数
def Gaussian(avg, sigma, x):
    n = len(avg)
    res = 1/(math.pow(2 * math.pi, n/2) *
             math.pow(np.linalg.det(sigma), 1/2)) * \
          math.exp(-1/2 * np.dot(np.dot((x - avg).T, np.linalg.inv(sigma)), (x - avg)))
    return res

class MOG:
    def __init__(self):
        self.data = []
        self.classNum = 0
        #聚类中心
        self.classCenter = []
        #聚类协方差矩阵
        self.classSigma = []
        #聚类均值向量
        self.classAvg = []
        #混合占比
        self.classAlpha = []

    def Train(self, classNum = 3, turn = 10000, data = []):
        self.data = data
        self.classNum = classNum
        for index in range(classNum):
            #初始化混合占比
            self.classAlpha.append(1/classNum)
            #随机使用classNum个变量当作初始平均向量
            randArr = data
            random.shuffle(randArr)
            size = len(data[0])
            for i in range(classNum):
                self.classAvg.append(np.array(randArr[i], dtype="float64").reshape([size, 1]))
            #初始协方差矩阵（维度是属性数量）
            matrix = np.zeros([size, size], dtype="float64")
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i][j] = 0.1
            self.classSigma.append(matrix)
        for t in range(turn):
            gama = []
            p = []
            for i in range(self.classNum):
                pi = []
                for j in range(len(data)):
                    pji = Gaussian(self.classAvg[i], self.classSigma[i],
                                   np.array(data[j], dtype="float64").reshape(size, 1))
                    pi.append(pji)
                p.append(pi)
            for j in range(len(data)):
                sumBase = 0
                gamaj = []
                for i in range(self.classNum):
                    sumBase += self.classAlpha[i] * \
                               Gaussian(self.classAvg[i], self.classSigma[i],
                                        np.array(data[j], dtype="float64").reshape(size, 1))
                for i in range(self.classNum):
                    res = self.classAlpha[i] * \
                          Gaussian(self.classAvg[i], self.classSigma[i],
                                   np.array(data[j], dtype="float64").reshape(size, 1)) / sumBase
                    gamaj.append(res)
                gama.append(gamaj)
            for i in range(self.classNum):
                sumBase = 0
                for j in range(len(data)):
                    sumBase += gama[j][i]
                self.classAvg[i] = np.zeros([size, 1], dtype="float64")
                for j in range(len(data)):
                    self.classAvg[i] += gama[j][i] * np.array(data[j], dtype="float64").reshape(size, 1)
                self.classAvg[i] /= sumBase
                self.classSigma[i] = np.zeros([size, size], dtype="float64")
                for j in range(len(data)):
                    d = np.array(data[j], dtype="float64").reshape(size, 1)
                    self.classSigma[i] += gama[j][i] * np.dot((d - self.classAvg[i]), (d - self.classAvg[i]).T)
                self.classSigma[i] /= sumBase
                self.classAlpha[i] = 1/len(data) * sumBase

    def Predict(self, data):
        p = 0
        c = 0
        size = len(data)
        for i in range(self.classNum):
            res = self.classAlpha[i] * \
                  Gaussian(self.classAvg[i], self.classSigma[i], np.array(data, dtype="float64").reshape(size, 1))
            if res > p:
                p = res
                c = i
        return c


if __name__ == "__main__":
    mog = MOG()
    datas = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
             [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
             [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
             [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
             [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
             [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437],
             [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376],
             [0.725, 0.445], [0.446, 0.459]]
    mog.Train(turn=1000, classNum=3, data=datas)
    marks = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for da in datas:
        p= mog.Predict(da)
        plt.plot(da[0], da[1], marks[p], markersize=5)
    plt.show()