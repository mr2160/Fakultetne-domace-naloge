import numpy as np
import cv2
import a3_utils as ut
from matplotlib import pyplot as plt
import e2

def line(x, y):
    binsT = 300
    binsR = 300
    # diag = np.sqrt(binsT**2 + binsR**2)
    diag = 100

    out = np.zeros((binsR, binsT))    
    valueT = (np.linspace(-np.pi/2, np.pi/2, binsT))
    ros = np.arange(-diag, diag, step=(2 * diag) / binsR)
    
    
    ro = (x * np.cos(valueT)) + (y * np.sin(valueT))
    ro = np.floor(ro).astype(int)

    
    for i in range(binsT):
        index = np.argmin(np.abs(ros - ro[i]))
        if(index>0 and index<binsR-1):
            out[index, i] += 1
    return out

def hugh_find_lines(I, binsT, binsR):
    diag = np.sqrt(I.shape[0]**2 + I.shape[1]**2)

    out = np.zeros((binsR, binsT))    
    thetas = (np.linspace(-np.pi/2, np.pi/2, binsT))
    

    X, Y = np.nonzero(I)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        ro = (x * np.cos(thetas)) + (y * np.sin(thetas))
        binR = np.round((ro+diag) / (2*diag) * binsR)
        binR = binR.astype(int)

    
        for j in range(binsT):
            index = binR[j]
            if(index>0 and index<binsR):
                out[index, j] += 1
    return out, diag

def nonmax_box(I):
    out = np.copy(I)
    for i in range(I.shape[0]-1):
        for j in range(I.shape[1]-1):
            nbrs = [
                I[i-1,j-1],
                I[i-1,j],
                I[i-1,j+1],
                I[i,j-1],
                I[i,j+1],
                I[i+1,j-1],
                I[i+1,j],
                I[i+1,j+1],
            ]
            if(I[i,j] <= np.amax(nbrs)):
                out[i,j] = 0
    return out


def a():
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(line(10,10))
    plt.subplot(2,2,2)
    plt.imshow(line(30,60))
    plt.subplot(2,2,3)
    plt.imshow(line(50,20))
    plt.subplot(2,2,4)
    plt.imshow(line(80,90))
    plt.show()

def b():
    I = np.zeros((300,300))
    I[100,100] = 255
    I[70,100] = 255
    I[100,200] = 255

    Iol = cv2.imread("images/oneline.png")
    Iol = np.mean(Iol, axis=2)
    Iole, _ = e2.findedges(Iol, 200) 
    Irec = cv2.imread("images/rectangle.png")
    Irec = np.mean(Irec, axis=2)
    Irece, _ = e2.findedges(Irec, 200)

    hughSynth, _ = hugh_find_lines(I, 300,300)
    hughLine, _ = hugh_find_lines(Iole, 300,300)
    hughRec, _ = hugh_find_lines(Irece, 300,300)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(hughSynth)
    plt.subplot(1,3,2)
    plt.imshow(hughLine)
    plt.subplot(1,3,3)
    plt.imshow(hughRec)
    plt.show()

def d(t):
    Iol = cv2.imread("images/oneline.png")
    Iol = np.mean(Iol, axis=2)
    Iole, _ = e2.findedges(Iol, 200) 
    Irec = cv2.imread("images/rectangle.png")
    Irec = np.mean(Irec, axis=2)
    Irece, _ = e2.findedges(Irec, 200)

    hughLine, diagL = hugh_find_lines(Iole, 300,300)
    hughRec, diagR = hugh_find_lines(Irece, 300,300)

    
    hughLine = nonmax_box(hughLine)
    hughRec = nonmax_box(hughRec)
    hughLine[hughLine < t] = 0
    hughRec[hughRec < t+50] = 0
    Xol, Yol = np.nonzero(hughLine)
    Xrec, Yrec = np.nonzero(hughRec)
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.imshow(Iol)
    plt.xlim([0, Iol.shape[1]])
    plt.ylim([Iol.shape[0], 0])
    for i in range(len(Xol)):
        ro = (Xol[i]*2*diagL)/300 - diagL
        theta = Yol[i]*np.pi/300 - np.pi/2
        ut.draw_line(ro, theta, diagL)
    
    plt.subplot(1,2,2)
    plt.imshow(Irec)
    plt.xlim([0, Irec.shape[1]])
    plt.ylim([Irec.shape[0], 0])
    for i in range(len(Xrec)):
        ro = (Xrec[i]*2*diagR)/300 - diagR
        theta = Yrec[i]*np.pi/300 - np.pi/2
        ut.draw_line(ro, theta, diagR)
    plt.show()


def e():
    B = cv2.imread("images/bricks.jpg")
    B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    Bbw = np.mean(B, axis=2)

    P = cv2.imread("images/pier.jpg")
    P = cv2.cvtColor(P, cv2.COLOR_BGR2RGB)
    Pbw = np.mean(P, axis=2)

    Be, _ = e2.findedges(Bbw, 70)

    # Pe, Pd = e2.findedges2(Pbw, 20, 10, 0.25)
    # Pe = e2.non_max_suppression(Pe, Pd)

    Pb = cv2.GaussianBlur(Pbw, (3,3), 0).astype(np.uint8)
    Pe = cv2.Canny(Pb, 30, 60)
    

    Blines, diagB = hugh_find_lines(Be, 150, 150)
    Plines, diagP = hugh_find_lines(Pe, 300, 600)

    Blines = nonmax_box(Blines)
    Plines = nonmax_box(Plines)
    

    Bcoords = np.transpose(np.nonzero(Blines))
    Bvalues = [Blines[c[0],c[1]] for c in Bcoords]
    Bvalues, BcoordsX, BcoordsY = zip(*sorted(zip(Bvalues, Bcoords[:,0], Bcoords[:,1])))
    BcoordsX = BcoordsX[::-1]
    BcoordsY = BcoordsY[::-1]

    Pcoords = np.transpose(np.nonzero(Plines))
    Pvalues = [Plines[c[0],c[1]] for c in Pcoords]
    Pvalues, PcoordsX, PcoordsY = zip(*sorted(zip(Pvalues, Pcoords[:,0], Pcoords[:,1])))
    PcoordsX = PcoordsX[::-1]
    PcoordsY = PcoordsY[::-1]

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(B, cmap="gray")
    plt.xlim([0, Bbw.shape[1]])
    plt.ylim([Bbw.shape[0], 0])
    for i in range(10):
        x = BcoordsX[i]
        y = BcoordsY[i]
        ro = (x*2*diagB)/150 - diagB
        theta = y*np.pi/150 - np.pi/2
        ut.draw_line(ro, theta, diagB)

    plt.subplot(1,2,2)
    plt.imshow(P, cmap="gray")
    plt.xlim([0, Pbw.shape[1]])
    plt.ylim([Pbw.shape[0], 0])
    for i in range(10):
        x = PcoordsX[i]
        y = PcoordsY[i]
        ro = (x*2*diagP)/600 - diagP
        theta = y*np.pi/300 - np.pi/2
        ut.draw_line(ro, theta, diagP)
    plt.show()

# d(150)
e()