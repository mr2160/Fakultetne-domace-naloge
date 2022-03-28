import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

# a
def myhist3(image, nbins):
    hist = np.zeros((nbins, nbins, nbins))
    for i in range(image[:,0,0].size):
        for j in range(image[0,:,0].size):
            bin0 = int(image[i,j,0]*nbins/256)
            bin1 = int(image[i,j,1]*nbins/256)
            bin2 = int(image[i,j,2]*nbins/256)
            hist[bin0, bin1, bin2] +=1
    return hist/(np.sum(hist))
#b
def compare_histograms(metric, hist1, hist2):
    if(metric == "L2"):
        temp = (hist1-hist2)**2
        temp = np.sum(temp)
        return temp**(0.5)
    if(metric == "Chi"):
        e0 = 0.00000000001
        temp = (hist1-hist2)**2
        temp = temp / (hist1 + hist2 + e0)
        temp = np.sum(temp)
        return 0.5*temp
    if(metric == "Inter"):
        tmin = np.minimum(hist1, hist2)
        
        temp = np.sum(tmin)
        return 1-temp
    if(metric == "Hell"):
        temp = ((hist1**0.5)-(hist2**0.5))**2
        temp = (0.5*np.sum(temp))**0.5
        return temp

def c():
    I1 = cv2.imread('dataset/object_01_1.png')
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.imread('dataset/object_02_1.png')
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    I3 = cv2.imread('dataset/object_03_1.png')
    I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)

    hist1 = myhist3(I1, 8).reshape(-1).reshape(-1)
    hist2 = myhist3(I2, 8).reshape(-1).reshape(-1)
    hist3 = myhist3(I3, 8).reshape(-1).reshape(-1)

    metric = "L2"

    c12 = compare_histograms(metric, hist1, hist2)
    c13 = compare_histograms(metric, hist1, hist3)
    c11 = compare_histograms(metric, hist1, hist1)

    plt.figure(figsize=(10,7))

    plt.subplot(2,3,1)
    plt.imshow(I1)
    plt.subplot(2,3,2)
    plt.imshow(I2)
    plt.subplot(2,3,3)
    plt.imshow(I3)

    plt.subplot(2,3,4)
    plt.bar(range(8*8*8), hist1, width=6)
    plt.title(metric+" I1,I2:"+str(c12))
    plt.subplot(2,3,5)
    plt.bar(range(8*8*8), hist2,  width=6)
    plt.title(metric+" I1,I3:"+str(c13))
    plt.subplot(2,3,6)
    plt.bar(range(8*8*8), hist3,  width=6)
    plt.title(metric+" I1,I1:"+str(c11))
    plt.show()

    # Object 3 is much more similar on all metrics. Black is expressed most intensly (because its the background).

def retrieve(path, nbins):
    res=[]
    for filename in glob.glob(path+'*.png'):
        I = cv2.imread(filename)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        histI = myhist3(I, nbins).reshape(-1).reshape(-1)
        res.append([filename, histI])
    return res

def d(path):
    # pozor: grdo
    I = cv2.imread(path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    histI = myhist3(I, 8).reshape(-1).reshape(-1)
    histograms = retrieve('dataset/', 8)
    distances = np.zeros((len(histograms),2))
    for i in range(len(histograms)):
        dist = compare_histograms("L2", histI, histograms[i][1])
        distances[i] = [i, dist]
    
    ind = np.argsort( distances[:,1] ) 
    distances = distances[ind]
    
    plt.figure(figsize=(10,7))
    for i in range(6):
        path = histograms[int(distances[i][0])][0]
        print(path)
        It = cv2.imread(path)
        It = cv2.cvtColor(It, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 6, i+1)
        plt.imshow(It)
        plt.subplot(2, 6, i+7)
        disti = round(distances[i][1], 2)
        plt.title(str(disti))
        plt.bar(range(8*8*8), histograms[i][1], width=7)

    plt.show()
    plt.savefig("e1d")
    # There is no big differene between the techniques. The execution time is greatly affected by the number of bins

def e(path):
    I = cv2.imread(path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    histI = myhist3(I, 8).reshape(-1).reshape(-1)
    histograms = np.array(retrieve('dataset/', 8))
    distances = np.zeros((len(histograms),2))
    for i in range(len(histograms)):
        dist = compare_histograms("Chi", histI, histograms[i][1])
        distances[i] = [i, dist]
    
    
    ind = np.argsort( distances[:,1] ) 
    distancesSort = distances[ind]
    marks = distancesSort[:6,0].astype(np.uint8)
    print(marks)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(distances[:,1], markevery=marks, marker="o", mfc="none", label="points")
    plt.subplot(1,2,2)
    plt.plot(distancesSort[:,1], markevery=range(6), marker="o", mfc="none", label="points")
    plt.show()
    plt.savefig("e1e")

#c()
#e('dataset/object_30_2.png')
d('dataset/object_30_2.png')