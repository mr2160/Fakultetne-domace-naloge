import numpy as np
import cv2
from matplotlib import pyplot as plt
def gauss(sigma):
    base = np.indices((11,))[0] - 5
    temp = -(base**2) / (2*(sigma**2))
    temp = np.exp(temp)
    factor = 1 / (2*np.pi*sigma)**0.5
    res = factor*temp
    return res/sum(res)

def gaussdx(sigma):
    base = np.indices((11,))[0] - 5
    factor = -1/(((2*np.pi)**0.5)*(sigma**3))
    exp = -(base**2)/(2*(sigma**2))
    res = factor*base*np.exp(exp)
    return res/sum(np.abs(res))

def zapConv(im, ker1, ker2):
    res = cv2.filter2D(src=im, ddepth=-1, kernel=ker1)
    res = cv2.filter2D(src=res, ddepth=-1, kernel=ker2)
    return res

def nonmax(image, n):
    x,y=image.shape
    res = np.copy(image)
    res[0:n,:] = 0
    res[:,0:n] = 0
    res[x-n:x,:] = 0
    res[:,y-n:y] = 0
    for i in range(n,image.shape[0]-n):
        for j in range(n,image.shape[1]-n):
            okolica = image[i-n:i+n,j-n:j+n]
            if(np.any(okolica > image[i,j])):
                res[i,j]=0
    return res

def hessianCoor(image, sigma):
    #dx = np.array(gaussdx(sigma))[np.newaxis]
    dx=np.array([[1,0,-1]])

    image = cv2.GaussianBlur(image, (25,25), sigma)
    Ixx = zapConv(image, dx, dx)
    Iyy = zapConv(image, dx.T, dx.T)
    Ixy = zapConv(image, dx.T, dx)
    det = (Ixx*Iyy - Ixy*Ixy)*(np.power(sigma,4))
    det = nonmax(det, 15)
    det[det < 100] = 0
    return np.where(det > 0)

def a():
    c3 = hessianCoor(im1, 3)
    c6 = hessianCoor(im1, 6)
    c9 = hessianCoor(im1, 9)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(im1, cmap="gray")
    plt.plot(c3[1], c3[0], 'x')
    plt.subplot(1,3,2)
    plt.imshow(im1, cmap="gray")
    plt.plot(c6[1], c6[0], 'x')
    plt.subplot(1,3,3)
    plt.imshow(im1, cmap="gray")
    plt.plot(c9[1], c9[0], 'x')
    plt.show()


def harris(image, sigma, treshold):
    dx=np.array([[1,0,-1]])
    im= cv2.GaussianBlur(image, (51,51), sigma)
    # Z uporab gaussdx bi lahko dosegel glajenje in odvod istočasno, 
    # sem se malo igral z različnimi odvodi.
    Ix = cv2.filter2D(src=im, ddepth=-1, kernel=dx)
    Iy = cv2.filter2D(src=im, ddepth=-1, kernel=dx.T)
    m1 = (sigma**2)*cv2.GaussianBlur(Ix*Ix, (51,51), 1.6*sigma)
    m2 = (sigma**2)*cv2.GaussianBlur(Iy*Iy, (51,51), 1.6*sigma)
    m3 = (sigma**2)*cv2.GaussianBlur(Ix*Iy, (51,51), 1.6*sigma)
    det = m1*m2 - m3*m3
    trace = m1 + m2
    h = det - 0.06*trace*trace
    hh = h
    h[h<treshold]=0
    h = nonmax(h, 15)
    c = np.where(h > 0)
    return c, hh

def b():
    c3,h3 = harris(im2, 1, 100)
    # c6,h6 = harris(im2, 6, 100)
    # c9,h9 = harris(im2, 9, 100)

    plt.figure(figsize=(12,4))
    plt.imshow(im2, cmap="gray")
    plt.plot(c3[1], c3[0], 'x')
    
    plt.show()

im1 = cv2.imread("data/test_points.jpg")
im1 = np.mean(im1, axis=2)
im2 = cv2.imread("data/graf/graf1.jpg")
im2 = np.mean(im2, axis=2)
# im1 = np.zeros((1000,1000))
# im1[250:750,250:750]=255
b()














# coor1, det1 = hessianCoor(im1, 1)
# coor3, det3 = hessianCoor(im1, 3)
# coor6, det6 = hessianCoor(im1, 6)
# coor9, det9 = hessianCoor(im1, 9)
# plt.figure()
# plt.subplot(2,4, 1)
# plt.imshow(im1, cmap="gray")
# plt.plot(coor1[1], coor1[0], 'x')
# plt.subplot(2,4, 2)
# plt.imshow(im1, cmap="gray")
# plt.plot(coor3[1], coor3[0], 'x')
# plt.subplot(2,4, 3)
# plt.imshow(im1, cmap="gray")
# plt.plot(coor6[1], coor6[0], 'x')
# plt.subplot(2,4, 4)
# plt.imshow(im1, cmap="gray")
# plt.plot(coor9[1], coor9[0], 'x')
# plt.subplot(2,4,5)
# plt.imshow(det1, cmap="gray")
# plt.subplot(2,4,6)
# plt.imshow(det3, cmap="gray")
# plt.subplot(2,4,7)
# plt.imshow(det6, cmap="gray")
# plt.subplot(2,4,8)
# plt.imshow(det9, cmap="gray")
# plt.show()

