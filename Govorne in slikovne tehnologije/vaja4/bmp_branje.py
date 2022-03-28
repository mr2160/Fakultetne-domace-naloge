
import struct
import math
import numpy as np
import matplotlib.pyplot as plt

# For information regarding the BMP format, see:
# http://en.wikipedia.org/wiki/BMP_file_format




if __name__ == '__main__':
    fname = 'parrots.bmp'
    with open(fname, 'rb') as fp:
        # format glave - little endian ter št. bajtov na argument
        fmt = '<QIIHHHH'

        #BM
        fp.read(2)
        #velikost
        fp.read(4)
        #nule
        fp.read(4)
        #index prve vrednosti piksla
        index = fp.read(4)[::-1].hex()
        index = int(index, 16)
        #velikost DIB dela glave
        fp.read(4)
        #N, M
        N = fp.read(2)[::-1].hex()
        N = int(N, 16)
        M = fp.read(2)[::-1].hex()
        M = int(M, 16)
        #preostalo
        fp.read(index-22)

        slika = np.empty((M,N,3), dtype=int)
        for y in range(M-1, 0, -1):
            for x in range(0, N):
                slika[y][x][2] = int(fp.read(1).hex(), 16)
                slika[y][x][1] = int(fp.read(1).hex(), 16)
                slika[y][x][0] = int(fp.read(1).hex(), 16)
        plt.imshow(slika)
        plt.show()
        

        
        


        
        