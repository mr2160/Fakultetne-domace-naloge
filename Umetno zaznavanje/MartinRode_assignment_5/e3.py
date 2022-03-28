import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils as ut
from mpl_toolkits import mplot3d

def triangulate(pt1, pt2, P1, P2):
    pt1 = np.array([pt1[0], pt1[1], 1])
    pt2 = np.array([pt2[0], pt2[1], 1])
    matrika1 = np.array([[0, -pt1[2], pt1[1]], 
                        [pt1[2], 0, -pt1[0]], 
                        [-pt1[1], pt1[0], 0]])
    matrika2 = np.array([[0, -pt2[2], pt2[1]], 
                        [pt2[2], 0, -pt2[0]], 
                        [-pt2[1], pt2[0], 0]])
    prod1 = matrika1 @ P1
    prod2 = matrika2 @ P2
    A = np.stack([prod1[0,:], prod1[1,:],prod2[0,:], prod2[1,:]])
    
    u, d, v = np.linalg.svd(A)
    ev = v[-1]
    return ev/ev[-1]

I1 = cv2.imread("data/epipolar/house1.jpg")
I1 = np.mean(I1, axis=2)
I2 = cv2.imread("data/epipolar/house2.jpg")
I2 = np.mean(I2, axis=2)

P1 = ut.read_data("data/epipolar/house1_camera.txt")
P1 = P1.reshape((3,4))
P2 = ut.read_data("data/epipolar/house2_camera.txt")
P2 = P2.reshape((3,4))

points = np.array(ut.read_data("data/epipolar/house_points.txt"))
points = points.reshape((10,4))
points1 = points[:,:2]
points2 = points[:,2:4]

res = np.empty((4,))
for i in range(len(points1)):
    res = np.vstack((res, triangulate(points1[i], points2[i], P1, P2)))
res = res[1:,:3]

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(133, projection='3d')
i1 = fig.add_subplot(131)
i2 = fig.add_subplot(132)

i1.imshow(I1, cmap="gray")
i2.imshow(I2, cmap="gray")
for i, pt in enumerate(points1):
    i1.plot(pt[0], pt[1], 'r.')
    i1.text(pt[0], pt[1], str(i))
for i, pt in enumerate(points2):
    i2.plot(pt[0], pt[1], 'r.')
    i2.text(pt[0], pt[1], str(i))

T = np.array([[-1,0,0],[0,0,1],[0,-1,0]]) # transformation matrix
res = np.dot(res,T)
for i, pt in enumerate(res):
    ax.plot([pt[0]],[pt[1]],[pt[2]],'r.')
    ax.text(pt[0],pt[1],pt[2], str(i)) 
plt.show()