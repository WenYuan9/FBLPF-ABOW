import matplotlib.pyplot as plt
import numpy as np

from Fuc import *
from Fuc_FBLPF_ABOW import Remove
Noise_Data=np.load("Noise.npy")
Pure_Data=np.load("Pure.npy")

filtedata=Remove(Noise_Data, 200,)
plt.plot(Noise_Data,"k")
plt.plot(filtedata,"r")
plt.plot(Pure_Data,"y")
plt.show()
