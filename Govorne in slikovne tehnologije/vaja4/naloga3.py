import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# SHRANJEVANJE SLIK
# slika = cv2.imread('parrots.png')
# for i in range(0, 101):
#     cv2.imwrite('slike\parrot'+str(i)+'.jpg', slika, [cv2.IMWRITE_JPEG_QUALITY, i])

def bpp(pot):
    velikost = os.path.getsize(pot)
    visina, sirina, kanali = cv2.imread(pot).shape
    return 8*velikost/(visina*sirina)

def popacenost(pot):
    original = cv2.imread('parrots.png')
    popacena = cv2.imread(pot)
    visina, sirina, kanali = original.shape
    rmse = np.sum((original-popacena)**2)
    return np.sqrt(rmse/(visina*sirina))

BPP = np.zeros(101)
RMSE = np.zeros(101)
for i in range(0,101):
    pot = 'slike/parrot'+str(i)+'.jpg'
    BPP[i] = bpp(pot)
    RMSE[i] = popacenost(pot)
plt.plot(BPP, RMSE)
plt.show()

