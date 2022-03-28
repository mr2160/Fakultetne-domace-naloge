import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils as ut

# The disparity formula is d = (T*f)/Pz
# Disparity gets lower as Pz gets bigger

def disparity(pz):
    return (2.5*120) / pz

def b():
    pzs = np.linspace(0, 5, 21)
    ds = [disparity(pz*1000) for pz in pzs]
    plt.figure()
    plt.plot(pzs, ds)
    plt.xlabel("Z [m]")
    plt.ylabel("Disparity [mm]")
    plt.show()

b()
