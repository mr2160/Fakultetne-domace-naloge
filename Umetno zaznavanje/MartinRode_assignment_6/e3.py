import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import a6_utils as ut

def dualPCA(points):
    pL = len(points)
    m = len(points[0])
    mean = np.array([np.mean(points, axis=0)])
    meanM = np.repeat(mean, pL, axis=0)
    centered = points - meanM
    covariance = np.matmul(centered, centered.T)
    covariance = (1/(m - 1))*covariance
    u, s, v = np.linalg.svd(covariance)
    s = s + 1/1000000000000000
    matrix = np.sqrt(np.linalg.inv(np.diag(s)*(m-1)))
    U = (centered.T @ u) @ matrix
    
    return U, mean, covariance, s

I = cv2.imread("data/faces/1/001.png")
m = I.shape[0]
n = I.shape[1]
imgs = np.empty((1,m*n))
for img in glob.glob("data/faces/1/*"):
    I = cv2.imread(img)
    I = np.mean(I, axis=2)
    I = np.reshape(I, (1,m*n))
    imgs = np.vstack((imgs, I))
imgs = imgs[1::]

u, mean, covariance, s = dualPCA(imgs)

def b(): #the images represent base images of an image space in which these five dimensions are responsible for the most variance
    plt.figure(figsize=(9,3))
    for i in range(5):
        pc = u[:,i]
        pc = np.reshape(pc, (m,n))
        plt.subplot(1,5,i+1)
        plt.imshow(pc, cmap="gray")
    plt.show()

def b2():
    im1 = np.reshape(imgs[3], (m,n))
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(im1, cmap="gray")
    
    
    impc = u.T @ imgs[3].T
    impc[1] = 0
    imrec = u @ impc
    imrec = np.reshape(imrec, (m,n))
    
    imMod = imgs[3]
    imMod[4074] = 0
    imMod = np.reshape(imMod, (m,n))

    plt.subplot(1,3,2)
    plt.imshow(imMod, cmap="gray")

    plt.subplot(1,3,3)
    plt.imshow(imrec, cmap="gray")
    plt.show()

def c():
    I = cv2.imread("data/faces/1/001.png")
    m = I.shape[0]
    n = I.shape[1]
    imgs = np.empty((1,m*n))
    for img in glob.glob("data/faces/1/*"):
        I = cv2.imread(img)
        I = np.mean(I, axis=2)
        I = np.reshape(I, (1,m*n))
        imgs = np.vstack((imgs, I))
    imgs = imgs[9::]

    u, mean, covariance, s = dualPCA(imgs)

    
    
    
    plt.figure(figsize=(9,3))
    for i in range(7):
        j = 2**i
        impc = u.T @ imgs[3].T
        impc[j:64]=0
        imrec = u @ impc
        imrec = np.reshape(imrec, (m,n))
        plt.subplot(1,7,i+1)
        plt.imshow(imrec, cmap="gray")
        plt.title(j)
    plt.show()
c()


