import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time



# SHRANJEVANJE SLIK
# slika = cv2.imread('IMG_2303.png')
# for i in range(0, 10):
#      cv2.imwrite('slike/velika'+str(i)+'.png', slika, [cv2.IMWRITE_PNG_COMPRESSION, i])

def bpp(pot):
    velikost = os.path.getsize(pot)
    visina, sirina, kanali = cv2.imread(pot).shape
    return 8*velikost/(visina*sirina)

casi = np.zeros(10)
BPP = np.zeros(10)
for i in range(10):
    pot = 'slike/velika'+str(i)+'.png'
    t0 = time.time()
    im = cv2.imread(pot)
    t1 = time.time()
    casi[i] = t1 - t0
    BPP[i] = bpp(pot)

plt.plot(BPP, casi)
plt.show()

