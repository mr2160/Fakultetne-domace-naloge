import numpy as np
from python_speech_features import mfcc, delta
from scipy.io.wavfile import read
import sys, os


def mfcc_feats(sig, fs): # implementiraj
    z0 = mfcc(sig, fs, winstep=0.015)
    z1red = delta(z0, 1)
    z2red = delta(z1red, 1)
    znacilke = np.concatenate((z0, z1red, z2red), 1)
    return znacilke

def evklidska(x,y):
    sum = 0
    for i in range(0,39):
        sum += (x[i]-y[i])**2
    return np.sqrt(sum)

def dtw_dist(sigF, sigH):
    P = len(sigF)
    R = len(sigH)
    C = np.zeros((P, R))
    C[0,0]
    for i in range(0,P-1):
        for j in range(0,R-1):
            Cmin = min([C[i,j], C[i,j+1], C[i+1,j]])
            C[i+1,j+1] = Cmin + evklidska(sigF[i], sigH[j])
    return C[P-1,R-1]

def primerjaj(pot1, pot2):
    fs1, signal1 = read("posnetki/"+pot1+".wav")
    fs2, signal2 = read("posnetki/"+pot2+".wav")
    znacilke1 = mfcc_feats(signal1, fs1)
    znacilke2 = mfcc_feats(signal2, fs2)
    return dtw_dist(znacilke1, znacilke2)

def prepoznaj(posnetek):
    ukazi = ["desnoMoj", "levoMoj", "gorMoj", "dolMoj"]
    najboljsi = 'desnoMoj1'
    razdalja = primerjaj(posnetek, 'desnoMoj1')
    for el in ukazi:
        for i in range(1,5):
            trenutnaPot = el + str(i)
            if(trenutnaPot == posnetek):
                continue

            trenutnaRazdalja = primerjaj(posnetek, trenutnaPot)
            print(trenutnaPot + ":" + str(trenutnaRazdalja))
            
            if(trenutnaRazdalja < razdalja):
                najboljsi = trenutnaPot
                razdalja = trenutnaRazdalja
    return najboljsi


if __name__ == "__main__":
    fs, signal = read("posnetki/gor1.wav")
    dejanske_znacilke = mfcc_feats(signal, fs=16000)
    pravilne_znacilke = np.load("mfcc_test.npy")

    # primerjava oblike zaporedij vektorjev značilk - to bi se moralo ujemati
    #print("Pravilna oblika:", pravilne_znacilke.shape)
    #print("Dejanska oblika:", dejanske_znacilke.shape)
    # največje odstopanje med vašim izračunom 
    # in dejanskim rezultatom - 
    # to bi moralo biti blizu oz. enako 0
    print(prepoznaj("dol4"))
    #print("max. odstopanje:", 
     #     np.abs(pravilne_znacilke - dejanske_znacilke).max())