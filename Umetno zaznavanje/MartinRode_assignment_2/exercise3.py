import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from a2_utils import *
import math

def calc_gauss(sigma):
    size = 2*3*sigma + 1
    base = np.indices((int(size),))[0] - int(size/2)
    temp = -(base**2) / (2*(sigma**2))
    temp = np.exp(temp)
    factor = 1 / (2*np.pi*sigma)**0.5
    res = factor*temp
    return res/sum(res), base

def gaussfilter(image, sigma):
    kx, _ = calc_gauss(sigma)
    kx = np.array(kx)[np.newaxis]
    ky = kx.T

    temp = cv2.filter2D(src=image, ddepth=-1, kernel=kx)
    temp = cv2.filter2D(src=temp, ddepth=-1, kernel=ky)
    return temp

def a():
    I = cv2.imread('images/lena.png')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    Ibw = np.mean(I, axis=2)
    Ign = gauss_noise(Ibw, 200)
    Isp = sp_noise(Ibw, 0.05)
    Igns = gaussfilter(Ign, 2)
    Isps = gaussfilter(Isp, 2)
    plt.figure(figsize=(6,6))
    plt.subplot(2,2,1)
    plt.imshow(Ign, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(Isp, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(Igns, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(Isps, cmap='gray')
    plt.show()
    # Gaussian is removed much better.

def b():
    I1 = cv2.imread('images/museum.jpg')
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    Ibw = np.mean(I1, axis=2)
    Ibw = gaussfilter(Ibw, 2)
    Iblur = gaussfilter(Ibw, 2)
    Isharp = Ibw - 0.6*Iblur
    plt.figure(figsize=(9,3))
    plt.subplot(1,2,1)
    plt.imshow(Ibw[150:350, 150:350], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(Isharp[150:350, 150:350], cmap='gray')
    plt.show()

def simple_median(signal, width):
    n = int(width/2)
    res=np.zeros(signal.size)
    for i in range(n, signal.size-n):
        window=signal[i-n:i+1+n]
        res[i]=np.median(window, axis=0)
    return res

def c():
    
    signal = np.array([0,0,0,0.2,0.2,0.2,0.2,0.2,0.2,0.7,0.7,0.7,0.7,0.7,0.7,0,0,0,0,0,0.2,0.4,0.5])
    signalsp = np.array([0,0,2,0.2,0.2,0.2,0.2,2,0.2,0.7,0.7,0,0.7,0.7,0.7,0,0,0,0,2,0.2,0.4,0.5])
    signalmed = simple_median(signalsp, 3)
    
    kernel, _ = calc_gauss(1)
    signalgauss = cv2.filter2D(signalsp, ddepth=-1, kernel=kernel)
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,4,1)
    plt.plot(signal)
    plt.title("Original")
    plt.subplot(1,4,2)
    plt.plot(signalsp)
    plt.title("Noisy")
    plt.subplot(1,4,3)
    plt.plot(signalmed)
    plt.title("Median")
    plt.subplot(1,4,4)
    plt.plot(signalgauss)
    plt.title("Gauss")
    plt.show()
    #Median filter preforms better. Yes, the order matters. This is called a nonlinear filter

c()
