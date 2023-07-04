import scipy.stats
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import math
import random
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import rcParams
config = {"font.family":'Times New Roman', "font.size": 13,"lines.linewidth":0.8}
rcParams.update(config)
from sko.tools import func_transformer
from scipy import signal
import pandas as pd
import pywt
import copy
import matplotlib.pyplot as plt

def Remove_DWT(Data,level,wavename,Flag=False):
    if len(Data) % 2 != 0:
        Data = Data[:-1]
    wave = pywt.wavedec(Data, wavename, level=level)

    ya = pywt.waverec(np.multiply(wave, [0] + [1] * level).tolist(),wavename)

    return ya
def Determine_Pattern(Signal_List):
    N=len(Signal_List)
    Result=np.zeros((N,N))
    for temp_i in np.arange(N):
        for temp_j in np.arange(N):
            min_len=np.min([len(Signal_List[temp_i]),len(Signal_List[temp_j])])
            Result[temp_i,temp_j]=np.corrcoef(Signal_List[temp_i][:min_len],Signal_List[temp_j][:min_len])[0][1]
    Result=np.sum(Result,axis=0)
    Max_Index=Result.tolist().index(np.max(Result))
    return Max_Index
def CC(RealSignal,FiltedSignal): #文献41
    # return np.corrcoef(RealSignal,FiltedSignal)[0][1]
    return scipy.stats.pearsonr(RealSignal, FiltedSignal)[0]
def RMS(Data):
    return math.sqrt(sum([x ** 2 for x in Data]) / len(Data))
class SCA():
    def __init__(self, pop_size=50, n_dim=1, a=2, lb=[-100], ub=[100], max_iter=15, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.a = a # 感知概率
        self.func = func
        self.max_iter = max_iter  # max iter
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))] # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]
    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]
    def update(self, i):
        r1 = self.a - i * ((self.a) / self.max_iter)
        for j in range(self.pop):
            for k in range(self.n_dim):
                r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                r3 = 2 * random.uniform(0.0, 1.0)
                r4 = random.uniform(0.0, 1.0)
                if r4 < 0.5:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
                else:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # Function for fitness evaluation of new solutions
    def run(self):
        for i in range(self.max_iter):
            self.update(i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
def LL(Bad_Signal, error,myWavelet):
    if len(Bad_Signal)%2!=0:
        Bad_Signal=Bad_Signal[:-1]
    wave = pywt.wavedec(Bad_Signal, myWavelet, level=1)
    wave[1]=np.array([0]*len(wave[1]))
    ya = pywt.waverec(wave, myWavelet)
    Flag_before = CC(Bad_Signal,ya)

    level = 2
    while Flag_before > error:
        wave = pywt.wavedec(Bad_Signal, myWavelet, level=level)
        ya = pywt.waverec(np.multiply(wave, [1] + [0] * level).tolist(), myWavelet)
        Flag_before = CC(Bad_Signal,ya)
        level += 1
    if level==2:
        level = 1
    else:
        level = level - 2
    return level
class MyFilterBank(object):
    def Get_a(self,par):
        self.a = par
    @property
    def filter_bank(self,):
        a=self.a
        A = -(3 + a)
        B = (9 * a * a * a + 35 * a * a + 48 * a + 24) / (3 * a * a + 9 * a + 8)
        C = -8 * (1 + a) * (1 + a) * (1 + a) / (3 * a * a + 9 * a + 8)

        h0 = [0,1, 2 * (A + 1), 4 * (A + B + 1), 6 * A + 8 * B + 8 * C + 6, 8 * A + 8 * B + 16 * C + 6,6 * A + 8 * B + 8 * C + 6, 4 * (A + B + 1), 2 * (A + 1), 1]
        h0=np.array(h0)/16
        f0 = [0,1, 2 * a + 4, 8 * a + 7, 12 * a + 8, 8 * a + 7, 2 * a + 4, 1,0,0]

        # h0=[0,0.0217,-0.0098,-0.0932,0.2598,0.6430,0.2598,-0.0932,-0.0098,0.0217]
        # f0=[0,-0.0404,-0.0182,0.2904,0.5365,0.2904,-0.0182,-0.0404,0,0]
        k1 = (2 ** 0.5) / np.sum(h0)
        k2 = (2 ** 0.5) / np.sum(f0)
        h0 = k1 * np.array(h0)
        f0 = k2 * np.array(f0)
        # h0=self.LIST
        # # h0=[ 0.0033,-0.0126,-0.0062,0.0776, -0.0322,-0.2423,0.1384,0.7243,
        # #      0.6038 , 0.1601 ]
        # f0=[temp for index,temp in enumerate(h0[::-1])]
        # Hi_D
        h1=[(-1)**(index+1)*temp for index,temp in enumerate(f0)]
        # Hi_R
        f1=[(-1)**index*temp for index,temp in enumerate(h0)]

        return [h0, h1, f0, f1]
def LowPass(sig, fc, fs, butter_filt_order):
    B,A = signal.butter(butter_filt_order, np.array(fc)/(fs/2), btype='low')
    return signal.filtfilt(B, A, sig, axis=0)