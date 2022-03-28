import matplotlib.pyplot as plt
import numpy as np

F1 = np.array([300, 625, 300, 500, 220, 120, 400, 400])
F2 = np.array([750, 1450, 900, 2000,2500, 600, 2200, 1200])

for i in range(8):
    plt.plot(F1[i], F2[i], "o")
plt.ylim((0, 3000))
plt.xlim((1000, 0))
plt.ylabel("F2 [Hz]")
plt.xlabel("F1 [Hz]")

plt.legend("o a u E i O e @".split(" "))

plt.show()