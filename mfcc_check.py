from scipy import signal
from scipy.io.wavfile import write,read
import matplotlib.pyplot as plt
import math
import numpy as np
import IPython
from numpy import convolve
from scipy.signal import hamming
from scipy.ndimage.interpolation import shift
from scipy.linalg import toeplitz,inv
from scipy.fftpack import fft,fftshift,ifft,dct

digit=6
i=34
filename="./Extracted_Feats/"+str(digit)+"/"+str(i)+".npy"
zr=np.load(filename)
exc_samp=[i for i in range(64) if i not in [31,32,33,34]]
for d in range(10):
    dist=0
    count=1
    for i in range(64):
        filename="./Extracted_Feats/"+str(d)+"/"+str(i+1)+".npy"
        tarr=np.load(filename)
        for j in range(zr.shape[0]):
            for k in range(tarr.shape[0]):
                dist=dist+np.sum(np.abs(zr[j]-tarr[k]))
                count=count+1
    print (dist/count)