import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils as ut
from random import randrange
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
    desc1 = ut.simple_descriptors(im1, coords1, bins=16, radius=40, w=5)
    
    coords2, harr2 = harris(im2, 1, 100)
    coords2 = list(zip(coords2[1], coords2[0]))
    desc2 = ut.simple_descriptors(im2, coords2, bins=16, radius=40, w=5)
    
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

def harris(image, sigma, treshold):
    dx=np.array([[1,0,-1]])
    im= cv2.GaussianBlur(image, (11,11), sigma)
    # Z uporab gaussdx bi lahko dosegel glajenje in odvod istočasno, 
    # sem se malo igral z različnimi odvodi.
    Ix = cv2.filter2D(src=im, ddepth=-1, kernel=dx)
    Iy = cv2.filter2D(src=im, ddepth=-1, kernel=dx.T)
    m1 = (sigma**2)*cv2.GaussianBlur(Ix*Ix, (11,11), 1.6*sigma)
    m2 = (sigma**2)*cv2.GaussianBlur(Iy*Iy, (11,11), 1.6*sigma)
    m3 = (sigma**2)*cv2.GaussianBlur(Ix*Iy, (11,11), 1.6*sigma)
    det = m1*m2 - m3*m3
    trace = m1 + m2
    h = det - 0.06*trace*trace
    hh = h
    h[h<treshold]=0
    h = nonmax(h, 25)
    c = np.where(h > 0)
    return c, hh

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

def fundamental_matrix(points1, points2):
    points1, T1 = ut.normalize_points(points1)
    points2, T2 = ut.normalize_points(points2)

    l = len(points1)
    A = np.empty((0,9))
    for i in range(l):
        p1 = points2[i]
        p2 = points1[i]
        a = np.array([p1[0]*p2[0], p1[0]*p2[1], p1[0], p1[1]*p2[0], p1[1]*p2[1], p1[1], p2[0], p2[1],1])
        A = np.vstack((A, a))   
    u, d, v =  np.linalg.svd(A)
    F = np.reshape(v[8,:], (3,3))
    uf, df, vf = np.linalg.svd(F)
    df[2] = 0
    F = np.matmul(uf, np.matmul(np.diag(df), vf))
    return np.matmul(T2.T, np.matmul(F, T1))

def b():
    I1 = cv2.imread("data/epipolar/house1.jpg")
    I1 = np.mean(I1, axis=2)
    I2 = cv2.imread("data/epipolar/house2.jpg")
    I2 = np.mean(I2, axis=2)
    points = np.array(ut.read_data("data/epipolar/house_points.txt"))
    points = points.reshape((10,4))
    points1 = points[:,:2]
    points2 = points[:,2:4]
    F = fundamental_matrix(points1, points2)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(I1, cmap="gray")   
    plt.plot(points1[:,0],points1[:,1], "o")
    for point in points2:
        p = np.array([point[0], point[1], 1]).T
        l1 = np.matmul(F.T, p)
        ut.draw_epiline(l1, I1.shape[0], I1.shape[1])
    plt.subplot(1,2,2)
    plt.imshow(I2, cmap="gray")
    plt.plot(points2[:,0],points2[:,1], "o")
    for point in points1:
        p = np.array([point[0], point[1], 1]).T
        l2 = np.matmul(F, p)
        ut.draw_epiline(l2, I2.shape[0], I2.shape[1])
    plt.show()

def distance(l, p):
    stevec = abs(l[0]*p[0] + l[1]*p[1] + l[2])
    imenovalec = np.sqrt(l[0]*l[0] + l[1]*l[1])
    return stevec/imenovalec

def reprojection_error(F, p1, p2):
    l2 = np.matmul(F, p1)
    l1 = np.matmul(F.T, p2)
    d1 = distance(l1, p1)
    d2 = distance(l2, p2)
    return (d1+d2)/2

def c():
    points = np.array(ut.read_data("data/epipolar/house_points.txt"))
    points = points.reshape((10,4))
    points1 = points[:,:2]
    points2 = points[:,2:4]
    F = fundamental_matrix(points1, points2)

    res = 0
    for i in range(len(points1)):
        p1 = points1[i]
        p2 = points2[i]
        p1 = [p1[0], p1[1], 1]
        p2 = [p2[0], p2[1], 1]
        res += reprojection_error(F, p1, p2)
    print(res/len(points1))

def get_inliers(F, ptsS, ptsT, tres):
    inliersS = []
    inliersT = []
    for i in range(len(ptsS)):
        p1 = ptsS[i]
        p2 = ptsT[i]
        p1 = [p1[0], p1[1], 1]
        p2 = [p2[0], p2[1], 1]
        l2 = np.matmul(F, p1)
        l1 = np.matmul(F.T, p2)
        if reprojection_error(F, p1, p2) < tres:
            inliersS.append(p1)
            inliersT.append(p2)
    return np.array(inliersS), np.array(inliersT)

def ransac(ptsS, ptsT, k, tres):
    idx = np.random.randint(len(ptsS), size=8)
    sampleS = ptsS[idx,:]
    sampleT = ptsT[idx,:]
    Fm = fundamental_matrix(sampleS, sampleT)
    Ims, ImT = get_inliers(Fm, ptsS, ptsT, tres)
    for iteration in range(k-1):
        idx = np.random.randint(len(ptsS), size=8)
        sampleS = ptsS[idx,:]
        sampleT = ptsT[idx,:]
        F = fundamental_matrix(sampleS, sampleT)
        Is, It = get_inliers(F, ptsS, ptsT, tres)
        if len(Is) > len(Ims):
            Ims = Is
            Imt = It
            Fm = F
    return Fm, Ims, Imt 

def matches(I1, I2):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(I1, None)
    kp2, des2 = orb.detectAndCompute(I2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_points = np.array([m[0] for m in src_points])
    dst_points = np.array([m[0] for m in dst_points])
    return src_points, dst_points

def e():
    points = np.array(ut.read_data("data/epipolar/house_matches.txt"))
    points = points.reshape((168,4))
    points1 = points[:,:2]
    points2 = points[:,2:4]

    F, Is, It = ransac(points1, points2, 100, 0.5)

    # F = np.array([[-0.000000885211824, -0.000005615918803, 0.001943109518320],
    #     [0.000009392818702, 0.000000616883199, -0.012006630150442],
    #     [-0.001203084137613, 0.011037006977740, -0.085317335867129]])

    I1 = cv2.imread("data/epipolar/house1.jpg")
    I1 = np.mean(I1, axis=2)
    I2 = cv2.imread("data/epipolar/house2.jpg")
    I2 = np.mean(I2, axis=2)

    px = 2
    p1 = Is[px]
    p2 = It[px]
    p1 = np.array([p1[0], p1[1], 1]).T
    p2 = np.array([p2[0], p2[1], 1]).T

    l2 = F @ p1
    l1 = F.T @ p1 

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.imshow(I1, cmap="gray")
    plt.scatter(points1[:,0], points1[:,1], facecolors='none', edgecolors="r")
    plt.scatter(Is[:,0], Is[:,1], facecolors='none', edgecolors="g", linewidths=2)
    plt.scatter(p1[0], p1[1], facecolors='none', edgecolors="b", linewidths=3)
    

    plt.subplot(1,2,2)
    plt.imshow(I2, cmap="gray")
    plt.scatter(points2[:,0], points2[:,1], facecolors='none', edgecolors="r")
    plt.scatter(It[:,0], It[:,1], facecolors='none', edgecolors="g", linewidths=2)
    plt.scatter(p2[0], p2[1], facecolors='none', edgecolors="b", linewidths=3)
    ut.draw_epiline(l2, I2.shape[0], I2.shape[1])
    error = reprojection_error(F, p1, p2)
    plt.title("error:" + str(round(error, 4)))
    plt.show()

def f():
    I1 = cv2.imread("data/epipolar/house1.jpg")
    I1b = np.mean(I1, axis=2)
    I2 = cv2.imread("data/epipolar/house2.jpg")
    I2b = np.mean(I2, axis=2)
    ptsS, ptsT = matches(I1, I2)
    
    F, Is, It = ransac(ptsS, ptsT, 100, 0.5)

    px  = np.random.randint(len(Is), size=1)[0]
    p1 = Is[px]
    p2 = It[px]
    p1 = np.array([p1[0], p1[1], 1]).T
    p2 = np.array([p2[0], p2[1], 1]).T

    l2 = F @ p1
    l1 = F.T @ p1 

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.imshow(I1, cmap="gray")
    plt.scatter(ptsS[:,0], ptsS[:,1], facecolors='none', edgecolors="r")
    plt.scatter(Is[:,0], Is[:,1], facecolors='none', edgecolors="g", linewidths=2)
    plt.scatter(p1[0], p1[1], facecolors='none', edgecolors="b", linewidths=3)
    

    plt.subplot(1,2,2)
    plt.imshow(I2, cmap="gray")
    plt.scatter(ptsT[:,0], ptsT[:,1], facecolors='none', edgecolors="r")
    plt.scatter(It[:,0], It[:,1], facecolors='none', edgecolors="g", linewidths=2)
    plt.scatter(p2[0], p2[1], facecolors='none', edgecolors="b", linewidths=3)
    ut.draw_epiline(l2, I2.shape[0], I2.shape[1])
    error = reprojection_error(F, p1, p2)
    plt.title("error:" + str(round(error, 4)))
    plt.show()
f()