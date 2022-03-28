import numpy as np
import cv2
from matplotlib import pyplot as plt
import a4_utils as ut

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

    image = cv2.GaussianBlur(image, (41,41), sigma)
    Ixx = zapConv(image, dx, dx)
    Iyy = zapConv(image, dx.T, dx.T)
    Ixy = zapConv(image, dx.T, dx)
    det = (Ixx*Iyy - Ixy*Ixy)*(np.power(sigma,4))
    det = nonmax(det, 15)
    det[det < 100] = 0
    return np.where(det > 0)

def harris(image, sigma, treshold):
    dx=np.array([[1,0,-1]])
    im= cv2.GaussianBlur(image, (31,31), sigma)
    # Z uporab gaussdx bi lahko dosegel glajenje in odvod istočasno, 
    # sem se malo igral z različnimi odvodi.
    Ix = cv2.filter2D(src=im, ddepth=-1, kernel=dx)
    Iy = cv2.filter2D(src=im, ddepth=-1, kernel=dx.T)
    m1 = (sigma**2)*cv2.GaussianBlur(Ix*Ix, (31,31), 1.6*sigma)
    m2 = (sigma**2)*cv2.GaussianBlur(Iy*Iy, (31,31), 1.6*sigma)
    m3 = (sigma**2)*cv2.GaussianBlur(Ix*Iy, (31,31), 1.6*sigma)
    det = m1*m2 - m3*m3
    trace = m1 + m2
    h = det - 0.06*trace*trace
    hh = h
    h[h<treshold]=0
    h = nonmax(h, 15)
    c = np.where(h > 0)
    return c, hh

def hell(vec1, vec2):
    temp = ((vec1**0.5)-(vec2**0.5))**2
    temp = (0.5*np.sum(temp))**0.5
    return temp

def find_correspondences(desc1, desc2):
    l = desc1.shape[0]
    l2 = desc2.shape[0]
    resVal = np.full(l, 100).astype(np.float32)
    resInd = np.zeros(l)
    for i in range(l):
        for j in range(l2):
            dist = hell(desc1[i], desc2[j])
            if(dist < resVal[i]):
                resVal[i] = dist
                resInd[i] = j
    return resInd, resVal

def find_matches(im1, im2):

    coords1, harr1 = harris(im1, 1, 100)
    
    coords1 = list(zip(coords1[1], coords1[0]))
    desc1 = ut.simple_descriptors(im1, coords1, bins=16, radius=40, w=11)
    
    coords2, harr2 = harris(im2, 1, 100)
    coords2 = list(zip(coords2[1], coords2[0]))
    desc2 = ut.simple_descriptors(im2, coords2, bins=16, radius=40, w=11)
    
    corr1, dists = find_correspondences(desc1, desc2)
    corr2, _ = find_correspondences(desc2, desc1)
    print(len(corr1), len(corr2))
    coord1Res = []
    coord2Res = []
    
    for i in range(len(corr1)):
        if(corr2[int(corr1[i])]==i):
            coord1Res.append(coords1[i])
            coord2Res.append(coords2[int(corr1[i])])
    matches = np.linspace(0,len(coord1Res)-1,len(coord1Res))
    matches = list(enumerate(matches))
    return matches, coord1Res, coord2Res, dists

def b():
    A1 = cv2.imread("data/newyork/newyork1.jpg")
    A1 = np.mean(A1, axis=2).astype(np.float64)
    A2 = cv2.imread("data/newyork/newyork2.jpg")
    A2 = np.mean(A2, axis=2).astype(np.float64)
    
    maxes = np.maximum(list(A1.shape), list(A2.shape))
    A1 = np.pad(A1, [(0, maxes[0]-A1.shape[0]), (0, maxes[1]-A1.shape[1])], mode='constant')
    A2 = np.pad(A2, [(0, maxes[0]-A2.shape[0]), (0, maxes[1]-A2.shape[1])], mode='constant')
    
    coords1, harr1 = harris(A1, 1, 100)
    coords1 = list(zip(coords1[1], coords1[0]))
    desc1 = ut.simple_descriptors(A1, coords1, radius=60)

    coords2, harr2 = harris(A2, 1, 100)
    coords2 = list(zip(coords2[1], coords2[0]))
    desc2 = ut.simple_descriptors(A2, coords2, radius=60)
    corr, _ = find_correspondences(desc1, desc2)
    matches = list(enumerate(corr))
    print(len(coords1), len(coords2))
    ut.display_matches(A1, A2, coords1, coords2, matches)

def c():
    A1 = cv2.imread("data/newyork/newyork1.jpg")
    A1 = np.mean(A1, axis=2).astype(np.float64)
    A2 = cv2.imread("data/newyork/newyork2.jpg")
    A2 = np.mean(A2, axis=2).astype(np.float64)
    
    maxes = np.maximum(list(A1.shape), list(A2.shape))
    A1 = np.pad(A1, [(0, maxes[0]-A1.shape[0]), (0, maxes[1]-A1.shape[1])], mode='constant')
    A2 = np.pad(A2, [(0, maxes[0]-A2.shape[0]), (0, maxes[1]-A2.shape[1])], mode='constant')
    
    matches, pts1, pts2, _ = find_matches(A1, A2)
    print(pts1)
    ut.display_matches(A1, A2, pts1, pts2, matches)
c()