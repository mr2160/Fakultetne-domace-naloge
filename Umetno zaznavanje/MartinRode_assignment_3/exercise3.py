import numpy as np
import cv2
from matplotlib import pyplot as plt
import a3_utils as utils
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
    Im = np.zeros((100,100))
    Im[80, 90] = 255
    
    resRo = 300
    resTheta = 300
    diag = np.sqrt(100**2 + 100**2)
    
    Acc = np.zeros((resRo, resTheta))
    
    xy = np.where(Im==255)

    theta = np.deg2rad(np.arange(-90, 90, 180/resTheta))
    cosSpace = np.cos(theta)
    sinSpace = np.sin(theta)
    

    for i in range(resTheta):
        ro = xy[0]*cosSpace[i] + xy[1]*sinSpace[i]
        ro = (ro+diag) * resRo / (2*diag) 
        
        if(ro < resRo and ro > 0):
            Acc[int(ro), i] += 1
        
    plt.figure()
    plt.imshow(Acc)
    plt.show()


def hough_find_lines(image, resRo, resTheta, tresholdD, tresholdU):
    
    image = cv2.Canny(image.astype(np.uint8), tresholdD, tresholdU)
    
    diag = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    
    Acc = np.zeros((resRo, resTheta))

    coords = np.nonzero(image)

    thetas = np.linspace(-np.pi/2, np.pi/2, resTheta)
    # ros = np.arange(-diag, diag, (2*diag)/resRo)
    cosSpace = np.cos(thetas)
    sinSpace = np.sin(thetas)
    
    for j in range(len(coords[0])):
        x = coords[0][j]
        y = coords[1][j]
        ro = x*cosSpace + y*sinSpace
        ro = (ro+diag) * resRo / (2*diag) 
        for i in range(resTheta):
            if(ro[i] < resRo and ro[i] > 0):
                Acc[int(ro[i]), i] += 1

    return Acc


def b():
    
    I = cv2.imread("images/oneline.png")
    I2 = cv2.imread("images/rectangle.png")
    
    Im = np.zeros((100, 100))
    Im[80, 90] = 255
    Im[30, 75] = 255
    Im[30, 85] = 255
    Im[11, 11] = 255
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(hough_find_lines(Im, 300, 200, 30, 60))
    plt.subplot(1,3,2)
    plt.imshow(hough_find_lines(I, 300, 200, 30, 60))
    plt.subplot(1,3,3)
    plt.imshow(hough_find_lines(I2, 300, 200, 30, 60))
    plt.show()

def d():
    I = cv2.imread("images/oneline.png")
    acc = nonmax(hough_find_lines(I, 300, 200, 30, 60))
    tres = 100
    acc1 = acc
    acc[acc < tres] = 0
    coor = np.nonzero(acc)
    maxro = np.max(coor[0])

    plt.figure()
    plt.imshow(I)
    
    for i in range(len(coor[0])):
        ro = coor[0][i]
        theta = coor[1][i]
        utils.draw_line(ro, theta, maxro)
    plt.show()
d()