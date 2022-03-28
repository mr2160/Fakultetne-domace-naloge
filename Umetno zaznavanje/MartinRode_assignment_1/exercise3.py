import numpy as np
import cv2
from matplotlib import pyplot as plt

#a
# I = cv2.imread('images/mask.png')

# n = 5
# SE = np.ones((n,n), np.uint8)
# I_eroded = cv2.erode(I, SE)
# I_dilated = cv2.dilate(I, SE)
# I_opened = cv2.dilate(I_eroded, SE)
# I_closed = cv2.erode(I_dilated, SE)

# n = 8
# SE = np.ones((n,n), np.uint8)
# I_eroded2 = cv2.erode(I, SE)
# I_dilated2 = cv2.dilate(I, SE)

# plt.subplot(1,3,1)
# plt.imshow(I)
# plt.subplot(1,3,2)
# plt.imshow(I_eroded)
# plt.subplot(1,3,3)
# plt.imshow(I_opened)
# plt.show()
# erosion followed by dilation is opening
# dilation followed by erosion is closing

#b

def myhist(image, nbins):
    hist = np.zeros(nbins)
    image = image.reshape(-1)
    for pixel in image:
        bin = (pixel*nbins/255).astype(np.uint8)
        hist[bin]+=1
    return hist/(np.sum(hist))

def otsu(image):
    t_opt = -1
    var_max = -1
    hist = myhist(image, 256)
    for t in range(256):
        l = hist[:t]
        d = hist[t:]
        l_mn = np.mean(l)
        d_mn = np.mean(d)
        var = l.size*d.size*((l_mn-d_mn)**2)
        if var > var_max:
            var_max = var
            t_opt = t
    return t_opt

def open(image, SE):
    eroded = cv2.erode(image, SE)
    return cv2.dilate(eroded, SE)

def close(image, SE):
    dilated = cv2.dilate(image, SE)
    return cv2.erode(dilated, SE)

# B = cv2.imread('images/bird.jpg')
# B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
# Bbw = np.average(np.copy(B), axis=2).astype(np.uint8)
# Bm = (Bbw > otsu(Bbw)+10).astype(np.uint8)


# #SE = np.ones((13,13), np.uint8)
# SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
# Bm_clean = close(Bm, SE)
# plt.imshow(Bm_clean, cmap=plt.cm.gray)
# plt.show()

#c

def immask(image, mask):
    temp = np.dstack((mask, mask, mask))
    return image*temp
#d

Ie = cv2.imread('images/eagle.jpg')
Ie = cv2.cvtColor(Ie, cv2.COLOR_BGR2RGB)
Iebw = np.average(np.copy(Ie), axis=2).astype(np.uint8)
Iem = (Iebw < otsu(Iebw)).astype(np.uint8)
plt.imshow(immask(Ie, Iem))
plt.show()

# If you know the foreground is darker than the background you can just filp
# the > in line 83. If you don't know you could make an if statement that would
# flip the >/< sign, depending on which side of the treshold has less pixels (the object
# is typically smaller than the background)

#e

# Ic = cv2.imread('images/coins.jpg')
# Ic = cv2.cvtColor(Ic, cv2.COLOR_BGR2RGB)
# Icbw = np.average(np.copy(Ic), axis=2).astype(np.uint8)
# Icm = (Icbw < 243).astype(np.uint8)
# SE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
# Icm = close(Icm, SE1)
# output = cv2.connectedComponentsWithStats(Icm)

# res = np.copy(Ic)
# lbl = output[1]
# for i in range(1, output[0]):
#     if output[2][i, -1] > 700:
#         res[lbl == i, :] = [255, 255, 255]
        
# plt.imshow(res)
# plt.show()
