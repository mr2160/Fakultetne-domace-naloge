import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math
import lpc

def sinsum(f, a, T, fs):
    """Funkcija za sintezo sinusne vrste. Vhodni argumenti:
        f: seznam frekvenc frekvenčnih komponent sinusne vrste
        a: seznam ojačanj frekvenčnih komponent, v linearni skali
        T: dolžina sintetiziranega signala, v sekundah
        fs: željena vzorčna frekvenca sintetiziranega signala."""
    
    # definicija časovne osi, preko katere sintetiziramo signal
    t = np.arange(0, T, 1/fs).astype("float32")
    # inicializacija praznega arraya za sintetiziran signal
    signal = np.zeros_like(t)

    for frekvenca, amplituda in zip(f, a):
        trenutniSig = amplituda * np.sin(2*math.pi*frekvenca*t)
        signal += trenutniSig


    # TODO: implementiraj sintezo s sinusno vrsto

    # normalizacija signala na zalogo vrednosti [-1, 1]
    signal -= signal.mean()
    signal /= np.abs(signal).max()
    return signal

if __name__ == "__main__":
    # x = sinsum([220, 440, 880, 1760], 
    #              [1, 0.5, 0.25, 0.125], 
    #          1.00002, 
    #           44100)

    # x_prav = np.load("sinsum_test.npy")
    
    # #oblika dejanskega in pravilnega signala
    # print(x.shape, x_prav.shape)
    # #
    # # maksimalno odstopanje
    # print(np.abs(x - x_prav).max())

    # plt.plot(x_prav)
    # plt.plot(x)
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    
    # For1 = [110, 520, 220, 200, 130]
    # Amp1 = [47.1, 47, 47, 47, 46.7]
    # For2 = [730, 1940, 2300, 600, 300]
    # Amp2 = [46.6, 46.9, 45.5, 46.3, 46.8]

    # For1 = [850, 610, 240, 500, 250]
    # Amp1 = [47.1, 47, 47, 47, 46.7]
    # For2 = [1610, 1900, 2400, 700, 595]
    # Amp2 = [46.6, 46.9, 45.5, 46.3, 46.8]
    # x = np.array([])

    # For1 = [850, 610, 240, 500, 250]
    # Amp1 = [47.1, 47, 47, 47, 46.7]
    # For2 = [1610, 1900, 2400, 700, 595]
    # Amp2 = [46.6, 46.9, 45.5, 46.3, 46.8]
    # x = np.array([])

    # for f1, f2, a1, a2 in zip(For1, For2, Amp1, Amp2):
    #     y = sinsum([f1, f2],
    #                 [a1, a2],
    #                 0.5,
    #                 44100)
    #     x = np.append(x, y)
    # wavfile.write("SamoglasnikiTeoretični.wav", 44100, x.astype(np.float32))


    fs, signal = wavfile.read("govorMoj.wav")
    
    x = np.array([])
    i=0
    while(i <= len(signal)-200):
        kos = signal[i:i+200]
        formanti, amplitude = lpc.lpc_okno(kos, 18, fs)
        y = sinsum(formanti,
                    amplitude,
                    100/fs,
                    16000)
        x = np.append(x, y)
        i += 100
    wavfile.write("govorMoj_synth.wav", 16000, x.astype(np.float32)) 


