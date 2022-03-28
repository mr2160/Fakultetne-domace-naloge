import numpy as np
import cv2
from matplotlib import pyplot as plt

def normalize_points(P):
	# P must be a Nx2 vector of points
	# first coordinate is x, second is y

	# returns: normalized points in homogeneous coordinates and 3x3 transformation matrix

	mu = np.mean(P, axis=0) # mean
	scale = np.sqrt(2) / np.mean(np.sqrt(np.sum((P-mu)**2,axis=1))) # scale
	T = np.array([[scale, 0, -mu[0]*scale],[0, scale, -mu[1]*scale],[0,0,1]]) # transformation matrix
	P = np.hstack((P,np.ones((P.shape[0],1)))) # homogeneous coordinates
	res = np.dot(T,P.T).T
	return res, T

def draw_epiline(l,h,w):
	# l: line equation (vector of size 3)
	# h: image height
	# w: image width
	x0, y0 = map(int, [0, -l[2]/l[1]])
	x1, y1 = map(int, [w-1, -(l[2]+l[0]*w)/l[1]])

	plt.plot([x0,x1],[y0,y1],'r')

	plt.ylim([0,h])
	plt.gca().invert_yaxis()

def warpTwoImages(img1, img2, H):
	'''warp img2 to img1 with homography H'''
	# copied off stackoverflow: https://stackoverflow.com/a/20355545
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
	return result

def get_grid():
	objectPoints= []
	grid_size = 0.035 # 3.5 cm
	rows, cols = 4, 11

	for i in range(cols):
		for j in range(rows):
			objectPoints.append( (i*grid_size, (2*j + i%2)*grid_size, 0) )

	objectPoints = np.array(objectPoints).astype('float32')

	return objectPoints

def read_data(filename):
	# reads a numpy array from a text file
	with open(filename) as f:
		s = f.read()

	return np.fromstring(s, sep=' ')

def gaussdxn(n):
    k = np.arange(-n, n + 1)
    sigma = n / 6
    k = -((1 / np.sqrt(np.pi * 2) * sigma ** 3)) * k * np.exp(-((k ** 2) / (2 * sigma ** 2)))
    k /= np.sum(np.abs(k))
    return np.array([k])


def gaussn(n):
    k = np.arange(-n, n + 1)
    sigma = n / 6
    k = np.exp(-(k ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.array([k])


def simple_descriptors(I, pts, bins=16, radius=40, w=11):
    g = gaussn(w)
    d = gaussdxn(w)

    Ix = cv2.filter2D(I, cv2.CV_64F, g.T)
    Ix = cv2.filter2D(Ix, cv2.CV_64F, d)

    Iy = cv2.filter2D(I, cv2.CV_64F, g)
    Iy = cv2.filter2D(Iy, cv2.CV_64F, d.T)

    Ixx = cv2.filter2D(Ix, cv2.CV_64F, g.T)
    Ixx = cv2.filter2D(Ixx, cv2.CV_64F, d)

    Iyy = cv2.filter2D(Iy, cv2.CV_64F, g)
    Iyy = cv2.filter2D(Iyy, cv2.CV_64F, d.T)

    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    mag = np.floor(mag * ((bins - 1) / np.max(mag)))

    feat = Ixx + Iyy
    feat += abs(np.min(feat))
    feat = np.floor(feat * ((bins - 1) / np.max(feat)))

    desc = []

    for y, x in pts:
        minx = max(x - radius, 0)
        maxx = min(x + radius, I.shape[0])
        miny = max(y - radius, 0)
        maxy = min(y + radius, I.shape[1])
        r1 = mag[minx:maxx, miny:maxy].reshape(-1)
        r2 = feat[minx:maxx, miny:maxy].reshape(-1)

        a = np.zeros((bins, bins))
        for m, l in zip(r1, r2):
            a[int(m), int(l)] += 1

        a = a.reshape(-1)
        a /= np.sum(a)

        desc.append(a)

    return np.array(desc)
