from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np

Vfrekvenca, data = read('govorMoj.wav')
print(Vfrekvenca)
dolzina = data.shape[0] / Vfrekvenca
print("%.3f s" % dolzina)
print(type(data[0]))
print("[%d, %d]" % (min(data), max(data)))

window_size = 0.050
NFFT = (int) (window_size * Vfrekvenca)

plt.specgram(data, NFFT, Vfrekvenca, noverlap=16)

plt.title("Spektrogram")
plt.xlabel("Čas[s]")
plt.ylabel("Frekvenca[Hz]")
plt.show()