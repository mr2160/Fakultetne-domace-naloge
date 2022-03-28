import numpy as np
import cv2
from matplotlib import pyplot as plt
import a4_utils as ut
import e2 as e2

# p3 translation, p1 scale, p2 rotation
#  
def homography(ptsS, ptsT):
     l = len(ptsS)
     A = np.empty((0,9))
     for i in range(l):
          pts = ptsS[i]
          ptt = ptsT[i]
          a1 = np.array([pts[0],pts[1],1,0,0,0,-ptt[0]*pts[0],-ptt[0]*pts[1],-ptt[0]])
          a2 = np.array([0,0,0,pts[0],pts[1],1,-ptt[1]*pts[0],-ptt[1]*pts[1],-ptt[1]])
          A = np.vstack((A, a1))
          A = np.vstack((A, a2))
     u, s, vh = np.linalg.svd(A)
     h = (vh[8,:])/(vh[8,:][-1])
     h = np.reshape(h, (3,3))
     return h, A

def b():
     im1 = cv2.imread("data/newyork/newyork1.jpg")
    #  im1 = cv2.imread("data/graf/graf1.jpg")
     im1 = np.mean(im1, axis=2)
     im2 = cv2.imread("data/newyork/newyork2.jpg")
    #  im2 = cv2.imread("data/graf/graf2.jpg")
     im2 = np.mean(im2, axis=2)
     points = ut.read_data("data/newyork/newyork.txt")
    #  points = ut.read_data("data/graf/graf.txt")
     points = np.reshape(points, (4,4))
     h, A = homography(points[:,0:2], points[:,2:4])
     print(h)
     imWarp = cv2.warpPerspective(im1, h, im1.shape)
     plt.figure()
     plt.subplot(1,2,1)
     plt.imshow(imWarp, cmap="gray")
     plt.subplot(1,2,2)
     plt.imshow(im2, cmap="gray")
     plt.show()
     #ut.display_matches(im1, im2, points[:,0:2], points[:,2:4], [(0,0),(1,1),(2,2),(3,3)])

def c():
     A1 = cv2.imread("data/newyork/newyork1.jpg")
     A1 = np.mean(A1, axis=2).astype(np.float64)
     A2 = cv2.imread("data/newyork/newyork2.jpg")
     A2 = np.mean(A2, axis=2).astype(np.float64)

     matches, pts1, pts2, dists = e2.find_matches(A1, A2)
     dists, matchesx, pts1x, pts2x = zip(*sorted(zip(dists, matches, pts1, pts2)))
     ptsS = []
     ptsT = []
     print(matchesx[:20])
     for s,t in matchesx[:20]:
          pt1 = list(pts1x[int(s)])
          pt2 = list(pts2x[int(t)])
          ptsS.append(pt1)
          ptsT.append(pt2)
     h, A = homography(ptsS, ptsT)
     print(h)
     imWarp = cv2.warpPerspective(A1, h, A1.shape)
     plt.figure()
     plt.subplot(1,2,1)
     plt.imshow(imWarp, cmap="gray")
     plt.subplot(1,2,2)
     plt.imshow(A2, cmap="gray")
     plt.show()
     #ut.display_matches(A1, A2, list(pts1x), list(pts2x), list(matchesx[:30]))
c()