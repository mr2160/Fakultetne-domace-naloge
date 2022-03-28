import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# namestitev ustrezne verzije knjižnjice openCV:
#pip uninstall opencv-python
#pip install -U opencv-contrib-python==3.4.2.16

def najdi_sifte(slika, ime):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(slika, None)
    koordin = np.zeros((len(kp),2))
    for i in range(len(kp)):
        koordin[i][0] = kp[i].pt[0]
        koordin[i][1] = kp[i].pt[1]
    koordin = koordin.astype(int)
    for tocka in koordin:
        slika1 = cv2.circle(slika,tuple(tocka),1,(0,0,255))
    cv2.imshow(ime, slika1)
    cv2.waitKey(1000)
    

def pariDeskriptorjev(des1, des2):
    # funkcija, ki izračuna pare deskriptorjev
    dists = np.zeros((des1.shape[0], des2.shape[0]), dtype="float64")
    # TODO: napolni matriko razdalj

    pari = []
    #TODO: za vsak deskriptor v des1 poišči najbližjega v des2 ter ga označi za
    #      par, če je razmerje med prvim in drugim najbližjim manjše od 0.25.
    #      Pare podaj z indeksi vrstic v des1 in des2 - npr. če sta par
    #      prvi deskriptor v des1 in četrti deskriptor v des2:
    #      pari.append((0, 3))
    return pari



def matchSift(fn1, fn2):
    im1 = cv2.imread(fn1)
    im2 = cv2.imread(fn2)
    kp1, des1 = najdi_sifte(im1)
    kp2, des2 = najdi_sifte(im2)

    pari = pariDeskriptorjev(des1, des2)

    stitched = np.concatenate((im1[:, :, ::-1], im2[:, :, ::-1]), axis=1)
    plt.imshow(stitched)
    for par in pari:
        i0, i1 = par
        x0, y0 = kp1[i0]
        x1, y1 = kp2[i1]
        x1 += im1.shape[1]

        plt.plot([x0, x1], [y0, y1])
    plt.show()

    pts1_matched = []
    pts2_matched = []
    for par in pari:
        i, j = par
        pts1_matched.append(kp1[i])
        pts2_matched.append(kp2[j])

    pts1_matched = np.array(pts1_matched)
    pts2_matched = np.array(pts2_matched)
    return pts1_matched, pts2_matched

if __name__ == "__main__":
    # TODO: uporabi najdene pare točk za določitev transformacijskih matrik
    #       in šivanje slik
    for i in range(1,10):
        for stran in ["left", "right"]:
            slika = cv2.imread('slike/par'+str(i)+'_'+stran+'.jpg')
            najdi_sifte(slika, str(i)+stran)
    cv2.waitKey(0)
    pass