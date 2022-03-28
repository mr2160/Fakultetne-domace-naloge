import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from a2_utils import read_data
import math

def simple_convolution(I, k):
    N = int((k.size-1)/2)
    res = np.zeros(I.size+(2*N))
    for i in range(N, I.size-N):
        tsum=0
        for j in range(k.size):
            tsum += I[i-j]*k[j]  
        res[i] = tsum
    return res

def b():
    kernel = read_data('kernel.txt')
    signal = read_data('signal.txt')
    plt.figure()
    plt.plot(simple_convolution(signal, kernel))
    plt.plot(kernel)
    plt.plot(signal)

    
    dest = cv2.filter2D(src=signal, ddepth=-1, kernel=kernel)
    plt.figure()
    plt.plot(dest)
    plt.plot(kernel)
    plt.plot(signal)
    plt.show()

    #The kernel is a Gaussian. The sum of elements is (really close to) 1. The effect is a general smoothing of the signal


def calc_gauss(sigma):
    size = 2*3*sigma + 1
    base = np.indices((int(size),))[0] - int(size/2)
    temp = -(base**2) / (2*(sigma**2))
    temp = np.exp(temp)
    factor = 1 / (2*np.pi*sigma)**0.5
    res = factor*temp
    return res/sum(res), base

def d():    
    plt.figure()

    kernel, base = calc_gauss(1)
    plt.plot(base, kernel)
    kernel, base = calc_gauss(2)
    plt.plot(base, kernel)
    kernel, base = calc_gauss(3)
    plt.plot(base, kernel)
    kernel, base = calc_gauss(4)
    plt.plot(base, kernel)
    kernel, base = calc_gauss(5)
    plt.plot(base, kernel)
    plt.show()
    plt.savefig("e2d")

def e():
    signal = read_data('signal.txt')
    k1, _ = calc_gauss(2)
    k2 = np.array([0.1, 0.6, 0.4])

    signal1 = cv2.filter2D(src=signal, ddepth=-1, kernel=k1)
    signal1 = cv2.filter2D(src=signal1, ddepth=-1, kernel=k2)

    signal2 = cv2.filter2D(src=signal, ddepth=-1, kernel=k2)
    signal2 = cv2.filter2D(src=signal2, ddepth=-1, kernel=k1) 

    k3 = cv2.filter2D(src=k1, ddepth=-1, kernel=k2)
    signal3 = cv2.filter2D(src=signal, ddepth=-1, kernel=k3)
    
    plt.figure(figsize=(9,3))
    plt.subplot(1,4,1)
    plt.plot(signal)
    plt.title("original")
    plt.subplot(1,4,2)
    plt.plot(signal1)
    plt.title("*k1)*k2")
    plt.subplot(1,4,3)
    plt.plot(signal2)
    plt.title("*k2)*k2")
    plt.subplot(1,4,4)
    plt.plot(signal3)
    plt.title("*(k1*k2)")
    plt.show()
    plt.savefig("e2e")

e()