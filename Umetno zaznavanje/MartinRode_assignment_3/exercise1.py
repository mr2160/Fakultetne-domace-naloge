import numpy as np
import cv2
from matplotlib import pyplot as plt
W = 10
SIGMA = 0.7
def gauss(w, sigma):
    size = w
    base = np.indices((int(size),))[0] - int(size/2)
    temp = -(base**2) / (2*(sigma**2))
    temp = np.exp(temp)
    factor = 1 / (2*np.pi*sigma)**0.5
    res = factor*temp
    return res/sum(res)

def gaussdx(w, sigma):
    size = w
    base = np.indices((int(size),))[0] - int(size/2)

    factor = -1/(((2*np.pi)**0.5)*(sigma**3))
    exp = -(base**2)/(2*(sigma**2))
    res = factor*base*np.exp(exp)
    return res/sum(np.abs(res))

def zapConv(im, ker1, ker2):
    res = cv2.filter2D(src=im, ddepth=-1, kernel=ker1)
    res = cv2.filter2D(src=res, ddepth=-1, kernel=ker2)
    return res

def c():
    impulse = np.zeros((25,25))
    impulse[12,12]=255

    # filter2D actually performs correlation, not convolution - so we must flip the kernels
    g = np.array(gauss(100, 3))[np.newaxis]
    gT = g.T
    gdx = -np.array(gaussdx(100, 3))[np.newaxis]
    gdxT = gdx.T


    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(impulse, cmap="gray")
    plt.subplot(2,3,2)
    plt.imshow(zapConv(impulse, g, gdxT), cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(zapConv(impulse, gdx, gT), cmap="gray")
    plt.subplot(2,3,4)
    plt.imshow(zapConv(impulse, g, gT), cmap="gray")
    plt.subplot(2,3,5)
    plt.imshow(zapConv(impulse, gT, gdx), cmap="gray")
    plt.subplot(2,3,6)
    plt.imshow(zapConv(impulse, gdxT, g), cmap="gray")
    plt.show()

def frstDer(image):
    g = np.array(gauss(W, SIGMA))[np.newaxis]
    gdx = -np.array(gaussdx(W, SIGMA))[np.newaxis]

    Ix = zapConv(image, g.T, gdx)
    Iy = zapConv(image, g, gdx.T)
    return Ix, Iy

def scndDer(image):
    g = np.array(gauss(W, SIGMA))[np.newaxis]
    gdx = -np.array(gaussdx(W, SIGMA))[np.newaxis]

    Ix, Iy = frstDer(image)
    Ixx = zapConv(Ix, g.T, gdx)
    Iyy = zapConv(Iy, g, gdx.T)
    Ixy = zapConv(Iy, g, gdx.T)
    return Ixx, Iyy, Ixy
def mag(image):
    Ix, Iy = frstDer(image)
    mag = Ix**2 + Iy**2
    return mag**0.5

def idir(image):
    Ix, Iy = frstDer(image)
    
    return np.arctan2(Iy, Ix)

def d():
    I = cv2.imread('images/museum.jpg')
    I = np.mean(I, axis=2).astype(np.float64)
    Ix, Iy = frstDer(I)
    Ixx, Iyy, Ixy = scndDer(I)
    Imag = mag(I)
    Idir = idir(I)
    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.subplot(2,4,2)
    plt.imshow(Ix, cmap="gray")
    plt.title("Ix")
    plt.subplot(2,4,3)
    plt.imshow(Iy, cmap="gray")
    plt.title("Iy")
    plt.subplot(2,4,4)
    plt.imshow(Ixx, cmap="gray")
    plt.title("Ixx")
    plt.subplot(2,4,5)
    plt.imshow(Iyy, cmap="gray")
    plt.title("Iyy")
    plt.subplot(2,4,6)
    plt.imshow(Ixy, cmap="gray")
    plt.title("Ixy")
    plt.subplot(2,4,7)
    plt.imshow(Imag, cmap="gray")
    plt.title("Imag")
    plt.subplot(2,4,8)
    plt.imshow(Idir, cmap="gray")
    plt.title("Idir")
    plt.show()

d()

