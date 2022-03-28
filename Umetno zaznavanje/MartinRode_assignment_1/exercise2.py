import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread('images/bird.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
Ibw = np.average(np.copy(I), axis=2)

#a
# mask = Ibw > 50
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(mask, plt.cm.gray)
# plt.subplot(1,2,2)
# plt.imshow(Ibw, plt.cm.gray)
# plt.suptitle('exercise a')
# plt.show()

#b

def myhist(image, nbins):
    hist = np.zeros(nbins)
    image = image.reshape(-1)
    for pixel in image:
        bin = (pixel*nbins/255).astype(np.uint8)
        hist[bin]+=1
    return hist/(np.sum(hist))
# Histograms are normalised 

# for i in range(1, 4):
#     bins = np.random.randint(2, 255)
#     plt.subplot(1, 3, i)
#     plt.bar(range(bins), myhist(Ibw, bins))
#     plt.title(bins)
# plt.show()

#c
def myhist2(image, nbins):
    hist = np.zeros(nbins)
    min = np.min(image)
    max = np.max(image)
    image = image.reshape(-1)
    image = image - min
    for pixel in image:
        bin = (pixel*nbins/max).astype(np.uint8)
        hist[bin]+=1
    return hist/(np.sum(hist))

# plt.subplot(1,2,1)
# plt.bar(range(255), myhist2(Ibw, 255))
# plt.subplot(1,2,2)
# plt.bar(range(255), myhist(Ibw, 255))
# plt.show()

#d
plt.figure(figsize = (9,9))
c = 1
for i in range(3):
    M = cv2.imread('images/miza'+str(i)+'.jpg')
    Mbw = np.average(np.copy(M), axis=2)
    plt.subplot(3,4,c)
    plt.imshow(M)
    c +=1
    for j in range(1, 4):
        bins = j * 75
        plt.subplot(3, 4, c)
        plt.bar(range(bins), myhist(Mbw, bins))
        plt.title(str(i)+ ": " + str(bins))
        c += 1
plt.show()
#Through different lightings the shape of the histogram is preserved, but
# it moves to the left the darker it gets.

#e

def otsu(image):
    t_opt = -1
    var_max = -1
    hist = myhist(image, 255)
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

# I = cv2.imread('images/bird.jpg')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# Ibw = np.average(np.copy(I), axis=2)
# mask1 = Ibw > otsu(Ibw)

# I2 = cv2.imread('images/eagle.jpg')
# I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
# Ibw2 = np.average(np.copy(I2), axis=2)
# mask2 = Ibw2 > otsu(Ibw2)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(mask1, cmap= plt.cm.gray)
# plt.subplot(1,2,2)
# plt.imshow(mask2, cmap= plt.cm.gray)
# plt.show()