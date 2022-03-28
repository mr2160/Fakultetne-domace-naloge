import numpy as np
import cv2
from matplotlib import pyplot as plt

I = cv2.imread('images/umbrellas.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
Ibw = np.average(np.copy(I), axis=2)
Icut = I[200:350,130:280,1]

Iinvert = np.copy(I)
Iinvert[100:260, 220:420, 0:3] = 255 - Iinvert[100:260, 220:420, 0:3]
#Inverting is defined by subtracting the value  froom 255

Ired = np.copy(Ibw).astype(float)
Ired = (Ired*63)/255
Ired = Ired.astype(np.uint8)



plt.subplot(2, 2, 1)
plt.imshow(Ibw, cmap="gray")
plt.subplot(2, 2, 2)
plt.imshow(Icut, cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(Iinvert)
plt.subplot(2, 2, 4)
plt.imshow(Ired, cmap="gray", vmax=255)
# We can use different color maps for different data presented in image form - for example 'plasma' for spectrograms or gray for grayscale images
plt.show()