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
def findedges(image, sigma, theta):
    res = mag(image)
    res[res < theta] = 0
    return res

def nonmax(image):
    Idir = idir(image)
    res = mag(image)
    image = res
  
    for j in range(len(image[0])-1):
        for i in range(len(image)-1):
            #print(f"{i}, {j} of {image.shape}")
            
            angle = Idir[i,j] * 180/np.pi
            if(angle<0): angle+=180

            
            if((angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle < 180)):
                if(image[i,j] < image[i, j-1] and image[i,j] < image[i, j+1]):
                    res[i,j] = 0
            
            if(angle >= 22.5 and angle < 67.5):
                if(image[i,j] < image[i-1, j+1] or image[i, j] < image[i+1, j-1]):
                    res[i,j] = 0

            if(angle >= 67.5 and angle < 112.5):
                if(image[i,j] < image[i-1, j] or image[i, j] < image[i+1, j]):
                    res[i,j] = 0

            if(angle >= 112.5 and angle < 157.5):
                if(image[i,j] < image[i-1, j-1] or image[i, j] < image[i+1, j+1]):
                    res[i,j] = 0

    return res


def a():
    I = cv2.imread('images/museum.jpg')
    I = np.mean(I, axis=2)
    
    Itres = findedges(I, 0, 0)
    plt.figure()
    plt.imshow(mag(Itres), cmap="gray")
    plt.show()

def b():
    I = cv2.imread('images/museum.jpg')
    I = np.mean(I, axis=2)

    Itest = np.zeros(I.shape)
    Itest[:,200:210]= 255

    Itres = findedges(I, 0, 25)
    nonmaxima = nonmax(I)
    
    plt.figure()
    plt.imshow(nonmaxima, cmap="gray")
    plt.show()
