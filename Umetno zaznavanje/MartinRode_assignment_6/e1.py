import numpy as np
import cv2
from matplotlib import pyplot as plt
import a6_utils as ut



def eDist(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[0])**2)

def PCA(points):
    pL = len(points)
    mean = np.array([np.mean(points, axis=0)])
    meanM = np.repeat(mean, pL, axis=0)
    centered = points - meanM
    covariance = np.matmul(centered.T, centered)
    covariance = (1/(pL - 1))*covariance
    u, s, v = np.linalg.svd(covariance)
    return u, mean, covariance, s

points = ut.read_data("data/points.txt")
points = np.reshape(points, (int(points.size/2),2))
u, mean, covariance, s = PCA(points)
pc1 = u[0,:]*s[0]
pc2 = u[1,:]*s[1]
print(u)
def b():
    plt.figure()
    plt.plot(points[:,0], points[:,1], 'x')
    plt.plot(mean[0,0], mean[0,1], 'x', color="red")
    ut.drawEllipse(mean[0], covariance)
    plt.arrow(mean[0,0], mean[0,1], pc1[0], pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], pc2[0], pc2[1], color="g")
    plt.axis('equal')
    plt.show()

def c():
    sN = s / s[0]
    plt.figure()
    plt.bar([0,1],sN)
    plt.show()
    print(sN)
    #The first vector represents 83% of the variance

def d():
    meanM = np.repeat(mean, len(points), axis=0)
    centered = points - meanM
    pointsPC = u.T @ centered.T
    pointsPC[1,:]=0
    pointsRec = u @ pointsPC
    pointsRec = pointsRec.T + meanM
    
    
    plt.figure()
    plt.plot(pointsRec[:,0], pointsRec[:,1], 'x')
    plt.plot(mean[0,0], mean[0,1], 'x', color="red")
    ut.drawEllipse(mean[0], covariance)
    plt.arrow(mean[0,0], mean[0,1], pc1[0], pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], -pc1[0], -pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], pc2[0], pc2[1], color="g")
    plt.axis('equal')
    plt.show()
    #The data is projected onto the line defined by pc1 and the mean point

def e(points):
    x = np.array([3,6])
    c = points[0]
    d = eDist(x , c)
    for p in points:
        dCurr = eDist(p, x)
        if(dCurr < d):
            c = p
            d = dCurr
    print(c)

    plt.figure()
    plt.plot(points[:,0], points[:,1], 'x')
    plt.plot(mean[0,0], mean[0,1], 'x', color="red")
    plt.plot(x[0], x[1], 'x', color="red")
    plt.plot(c[0], c[1], 'x', color="green")
    ut.drawEllipse(mean[0], covariance)
    plt.arrow(mean[0,0], mean[0,1], pc1[0], pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], pc2[0], pc2[1], color="g")
    plt.axis('equal')
    plt.show()

    points = np.vstack((points, x))
    
    meanM = np.repeat(mean, len(points), axis=0)
    centered = points - meanM
    pointsPC = u.T @ centered.T
    pointsPC[1,:]=0
    pointsRec = u @ pointsPC
    pointsRec = pointsRec.T + meanM

    x = pointsRec[-1]
    c = pointsRec[0]
    d = eDist(x , c)
    for p in pointsRec[0:-1]:
        dCurr = eDist(p, x)
        if(dCurr < d):
            c = p
            d = dCurr
    print(c)

    plt.figure()
    plt.plot(pointsRec[:,0], pointsRec[:,1], 'x')
    plt.plot(x[0], x[1], 'x', color="red")
    plt.plot(c[0], c[1], 'x', color="green")
    plt.plot(mean[0,0], mean[0,1], 'x', color="red")
    ut.drawEllipse(mean[0], covariance)
    plt.arrow(mean[0,0], mean[0,1], pc1[0], pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], -pc1[0], -pc1[1], color="b")
    plt.arrow(mean[0,0], mean[0,1], pc2[0], pc2[1], color="g")
    plt.axis('equal')
    plt.show()


b()
#e(points)