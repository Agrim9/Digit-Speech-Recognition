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

accuracy=0
for speak in range(1,65):
    for digit in range(10):
        print("Analysis for "+str(digit))
        filename="./Extracted_Feats/"+str(digit)+"/"+str(speak)+".npy"
        zr=np.load(filename)
        forbidden=[int(speak/4)*4+1,int(speak/4)*4+2,int(speak/4)*4+3,int(speak/4)*4+4]
        exc_samp=[i for i in range(64) if i not in forbidden]
        pred=np.zeros(10)
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
            pred[d]=(dist/count)
            #print(str(d)+":"+str(pred[d]))
        predic=np.argmin(pred)
        if(predic==digit):
            accuracy=accuracy+1
        print("Predicted: "+ str(predic))
print(accuracy/640)