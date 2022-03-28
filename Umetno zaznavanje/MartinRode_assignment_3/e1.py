import numpy as np
import cv2
from matplotlib import pyplot as plt

def gaussdx(w, sigma):
    k = np.arange(-w, w + 1)
    k = ((1 / np.sqrt(np.pi * 2) * sigma ** 3)) * k * np.exp(-((k ** 2) / (2 * sigma ** 2)))
    k /= np.sum(np.abs(k))
    return np.array([k])

def gauss(w, sigma):
    k = np.arange(-w, w + 1)
    k = np.exp(-(k ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.array([k])

def zapConv(im, ker1, ker2):
    res = cv2.filter2D(src=im, ddepth=-1, kernel=ker1)
    res = cv2.filter2D(src=res, ddepth=-1, kernel=ker2)
    return res

def partials(im, g, gd, partial):
    if partial == 'x':
        return zapConv(im, g.T, gd)
    elif partial == 'y':
        return zapConv(im, g, gd.T)
    elif partial == 'xx':
        temp = partials(im, g, gd, 'x')
        return zapConv(temp, g.T, gd)
    elif partial == 'yy':
        temp = partials(im, g, gd, 'y')
        return zapConv(temp, g, gd.T)
    elif partial == 'xy' or partial == 'yx':
        temp = partials(im, g, gd, 'x')
        return zapConv(temp, g, gd.T)
    else:
        return null

def gradient_magnitude(im, w, sigma):
    Ix = partials(im, gauss(w,sigma), gaussdx(w,sigma), 'x')
    Iy = partials(im, gauss(w,sigma), gaussdx(w,sigma), 'y')
    Imag = np.sqrt((Ix**2)+(Iy**2))
    Idir = np.arctan2(Iy, Ix)
    return Imag, Idir
 
def c():
    impulse = np.zeros((25,25))
    impulse[12,12] = 255

    G = gauss(10, 3)
    D = gaussdx(10, 3)

    ggt = zapConv(impulse, G, G.T)
    gdt = zapConv(impulse, G, D.T)
    dgt = zapConv(impulse, D, G.T)
    gtd = zapConv(impulse, G.T, D)
    dtg = zapConv(impulse, D.T, G)
    images = [impulse, gdt, dgt, ggt, gtd, dtg]

    plt.figure()
    for i in range(len(images)):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i], cmap="gray")
    plt.show()

def d():
    I = cv2.imread("images/museum.jpg")
    I = np.mean(I, axis=2)

    w = 10
    sigma=1

    Ix = partials(I, gauss(w,sigma), gaussdx(w,sigma), 'x')
    Iy = partials(I, gauss(w,sigma), gaussdx(w,sigma), 'y')
    Ixx = partials(I, gauss(w,sigma), gaussdx(w,sigma), 'xx')
    Ixy = partials(I, gauss(w,sigma), gaussdx(w,sigma), 'xy')
    Iyy = partials(I, gauss(w,sigma), gaussdx(w,sigma), 'yy')
    Imag, Idir = gradient_magnitude(I, w, sigma)


    images = [I, Ix, Iy, Imag, Ixx, Ixy, Iyy, Idir]
    names = ['I', 'Ix', 'Iy', 'Imag', 'Ixx', 'Ixy', 'Iyy', 'Idir']

    plt.figure(figsize=(12,6))
    for i in range(len(images)):
        plt.subplot(2,int(len(images)/2),i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(names[i])
    plt.show()

d()


