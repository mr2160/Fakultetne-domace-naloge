import struct
import math
# For information regarding the BMP format, see:
# http://en.wikipedia.org/wiki/BMP_file_format

def taDolgaEnacba(x):
    return 127.5*math.sin(20*math.pi*x) + 127.5

if __name__ == '__main__':
    # resolucija slike
    N, M = (512, 512)
    fname = 'test.bmp'
    with open(fname, 'wb') as fp:
        # format glave - little endian ter št. bajtov na argument
        fmt = '<QIIHHHH'
        # 'BM' == Windows bitmap identifier
        fp.write(b'BM' + struct.pack(fmt,
                                    # dolžina datoteke (piksli * 3 + glava)         
                                    N * M * 3 + 26,
                                    # indeks prve vrednosti piksla
                                    26, 
                                    # velikost DIB dela glave
                                    12,
                                    # resolucija po stolpcih in vrsticah
                                    N, M,
                                    # število slik
                                    1, 
                                    # bitna globina (biti/piksel)
                                    24))

        for y in range(0, N):
            for x in range(0, M):
                R = int(taDolgaEnacba((x/M + y/N)/2))
                G = int(taDolgaEnacba(y/N))
                B = int(taDolgaEnacba(x/M))
                fp.write(bytes([B, G, R]))

        # fp.write(bytes([0, 0, 0]))
        # fp.write(bytes([0, 0, 255]))
        # fp.write(bytes([0, 255, 0]))
        # fp.write(bytes([255, 0, 0]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
        # fp.write(bytes([127, 127, 127]))
