import numpy as np
import cv2
from matplotlib import pyplot as plt
import a6_utils as ut

def dualPCA(points):
    pL = len(points)
    m = len(points[0])
    mean = np.array([np.mean(points, axis=0)])
    meanM = np.repeat(mean, pL, axis=0)
    centered = points - meanM
    covariance = np.matmul(centered, centered.T)
    covariance = (1/(m - 1))*covariance
    u, s, v = np.linalg.svd(covariance)
    matrix = np.sqrt(np.linalg.inv(np.diag(s)*(m-1)))
    U = (centered.T @ u) @ matrix
    
    return U, mean, covariance, s

points = ut.read_data("data/points.txt")
points = np.reshape(points, (int(points.size/2),2))
u, mean, covariance, s = dualPCA(points)
pc1 = u[:,0]*s[0]
pc2 = u[:,1]*s[1]

meanM = np.repeat(mean, len(points), axis=0)
centered = points - meanM
pointsPC = u.T @ centered.T
pointsRec = u @ pointsPC
pointsRec = pointsRec.T + meanM
print(pointsRec)
plt.figure()
plt.plot(pointsRec[:,0], pointsRec[:,1], 'x')
plt.plot(mean[0,0], mean[0,1], 'x', color="red")
plt.axis('equal')
plt.show()