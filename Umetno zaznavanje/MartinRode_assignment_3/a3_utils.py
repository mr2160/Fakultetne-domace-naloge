import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_line(rho, theta, max_rho):
	a = np.cos(theta)
	b = np.sin(theta)

	x0 = a*rho
	y0 = b*rho

	x1 = int(x0 + max_rho*(-b))
	y1 = int(y0 + max_rho*(a))
	x2 = int(x0 - max_rho*(-b))
	y2 = int(y0 - max_rho*(a))

	plt.plot((y1,y2),(x1,x2),'r')

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
	Ix = partials(im, gauss(w, sigma), gaussdx(w,sigma), 'x')
	Iy = partials(im, gauss(w, sigma), gaussdx(w,sigma), 'y')
	Imag = ((Ix**2)+(Iy**2))**0.5
	Idir = np.arctan2(Iy, Ix)
	return Imag, Idir
 