import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

fs, signal = read("govorMoj.wav")

im = ["o", "a", "u", "E", "i", "O", "e", "@"]
t0 = [1.973, 2.974, 3.859, 4.441, 4.884, 5.252, 6.255, 6.542]
t1 = [2.097, 3.130, 3.892, 4.566, 5.010, 5.400, 6.320, 6.608]


# iz časovnega intervala izračunaj začetni in končni indeks izseka
for ime, tt0, tt1 in zip(im, t0, t1):
    i0 = (int) (tt0 * fs)
    i1 = (int) (tt1 * fs)

    izrez = signal[i0 : i1]
    NFFT = izrez.shape[0]
    I = np.fft.fft(izrez)[:len(izrez)//2]
    f_os = np.arange(0, fs/2, fs/NFFT)
    P = 20 * np.log10(np.abs(I))
    plt.plot(f_os, P)
    plt.title(ime)
    plt.grid(True)
    plt.show()