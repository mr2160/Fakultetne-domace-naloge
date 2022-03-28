import numpy as np
import cv2
import a3_utils as ut
from matplotlib import pyplot as plt

def findedges(I, t):
    Imag, Idir = ut.gradient_magnitude(I, 10, 0.25)
    Ie = np.zeros(Imag.shape)
    Ie[Imag >= t] = 1
    return Ie, Idir

def findedges2(I, t, n, sigma):
    Imag, Idir = ut.gradient_magnitude(I, n, sigma)
    Ie = np.zeros(Imag.shape)
    Ie[Imag >= t] = 1
    return Ie, Idir

def non_max_suppression(Imag, Idir):
    Idir = Idir*180/np.pi
    Idir[Idir < 0] += 180

    M,N = Imag.shape
    Ires = np.copy(Imag)
    for i in range(M-1):
        for j in range(N-1):
            angle = Idir[i,j]
            q = 0
            r = 0
            
            if(angle < 112.5 and angle >= 67.5):
                q = Ires[i+1,j]
                r = Ires[i-1,j]
            if(angle < 22.5 or angle > 157.5):
                q = Ires[i,j+1]
                r = Ires[i,j-1]
            if(angle <= 157.5 and angle > 112.5):
                q = Ires[i+1,j-1]
                r = Ires[i-1,j+1]
            if(angle < 67.5 and angle >= 22.5):
                q = Ires[i-1,j-1]
                r = Ires[i+1,j+1]
            
            if(Ires[i,j] <= q or Ires[i,j] <= r):
                Ires[i,j] = 0
            
    return Ires


def a():
    I = cv2.imread("images/museum.jpg")
    I = np.mean(I, axis=2)
    
    Ie, _ = findedges(I, 60)
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(Ie, cmap="gray")
    plt.show()

def b():
    I = cv2.imread("images/museum.jpg")
    I = np.mean(I, axis=2)

    # I = np.zeros((25,25))
    # for i in range(1,24):
    #     I[25-i,i] = 255
    #     I[24-i,i] = 255
    
    Imag, Idir = findedges(I, 60)
    Inm = non_max_suppression(Imag, Idir)
    images = [I, Imag, Inm]
    names = ['I', 'Tresholded, t=50', 'Non-maxima, t=50']
    
    plt.figure(figsize=(15,5))
    for i in range(len(images)):
        
        plt.subplot(1,int(len(images)),i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(names[i])
    plt.show()
b()

