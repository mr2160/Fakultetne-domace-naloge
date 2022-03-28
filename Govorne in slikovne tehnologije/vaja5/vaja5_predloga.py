import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import reduce
from scipy import ndimage

def skin_color_filter(im):
    r, g, b = [im[:, :, c] for c in range(3)]
    mask = reduce(np.logical_and, (r>95, g>40, b>20, abs(r-g)>15, r>g, r>b, np.max((r,g,b))-np.min((r,g,b))>15
            ))
    mask = mask.astype(np.uint8)
    return mask

def fill(mask):
    im_floodfill = mask.copy()
    
    h, w = mask.shape[:2]
    new_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, new_mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv
    return im_out

def dilate(mask):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def clean_skin_mask(mask):
    mask_mid=fill(mask)
    mask_out=dilate(mask_mid)
    return mask_mid

def rgb2gray(im):
    gray = np.mean(im, axis=2)
    return gray

def smooth_gray_image(im):
    return cv2.GaussianBlur(im, (7,7), 2)   

def gradient_approximation(im):
    sob = np.array([[-1 , 0 , 1] , [-2 , 0 , 2] , [-1 , 0 , 1] ])
    Gx = cv2.filter2D(im, -1, sob)
    Gy = cv2.filter2D(im, -1, np.transpose(sob))
    return np.sqrt(np.power(Gx,2)+np.power(Gy,2))/255
    

def mask_edges(edges, mask):
    return np.multiply(edges, mask)

def threshold_edges(edges):
    out = edges.copy()
    print(np.max(out))
    meja = 40
    out[out>=meja] = 255
    out[out<meja] = 0
    return out

def fit_ellipse(bin_edges):
    # vhod: binarna slika robov, maskiranih s področjem kože
    # izhod: parametri elipse
    points = []
    for x in range(bin_edges.shape[1]):
        for y in range(bin_edges.shape[0]):
            if bin_edges[y, x] > 0:
                points.append((x, y))
    
    points = np.array(points).astype("float32")
    ell = cv2.fitEllipse(points)
    return ell

def draw_ellipse(im, ell):
    # vhod: barvna slika in parametri elipse
    # izhod: slika z vrisano elipso
    centre, axes, angle = ell
    centre = (int(round(centre[0])), int(round(centre[1])))
    axes = (int(round(axes[0]/2)), int(round(axes[1]/2)))
    angle = int(round(angle))
    print(centre, axes, angle)

    im_in = im.astype("float64")
    face_det = cv2.ellipse(im_in, center=centre, axes=axes, angle=angle, 
                           startAngle=0.0, endAngle=360.0, color=(255, 0, 0),
                           thickness=3)
    return face_det.round().astype("uint8")

if __name__ == "__main__":
    # sestavi implementirane funkcije v celoten algoritem za zaznavanje obrazov,
    # opisan v navodilih
    slika = cv2.imread('slike\slika5.png')[:,:,::-1]
    slikaR = cv2.flip(slika, 1)
    maska = skin_color_filter(slikaR)
    maska1 = clean_skin_mask(maska)
    siva = rgb2gray(slikaR)
    siva1 = smooth_gray_image(siva)
    siva2= gradient_approximation(siva1)
    robovi = mask_edges(siva2, maska1)
    robovi_tres = threshold_edges(robovi)
    
    slikaZelipso = draw_ellipse(slikaR, fit_ellipse(robovi_tres))
    plt.imshow(slikaZelipso)
    plt.show()