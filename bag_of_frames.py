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
accuracy_perd=np.zeros(10)
for speak in range(1,65):
    print("Speaker "+ str(int((speak-1)/4)+1)+" Uterrence: "+ str(int(speak-1)%4+1))
    for digit in range(10):
        filename="./Extracted_Feats/"+str(digit)+"/"+str(speak)+".npy"
        zr=np.load(filename)
        forbidden=[int(speak/4)*4+1,int(speak/4)*4+2,int(speak/4)*4+3,int(speak/4)*4+4]
        exc_samp=[i for i in range(1,65) if i not in forbidden]
        #print(exc_samp)
        pred=np.zeros(10)
        for d in range(10):
            dist=0
            count=1
            for i in exc_samp:
                filename="./Extracted_Feats/"+str(d)+"/"+str(i)+".npy"
                tarr=np.load(filename)
                # Find all closest distances to the frames (Explained in Report)
                for j in range(zr.shape[0]):
                    min_dist_to_a_frame=0
                    # Find Closest Matching Frame
                    for k in range(tarr.shape[0]):
                        obt_dist=np.sum(np.abs(zr[j]-tarr[k]))
                        if(k==0):
                            min_dist_to_a_frame=obt_dist
                        else:
                            if(obt_dist<min_dist_to_a_frame):
                                min_dist_to_a_frame=obt_dist
                    dist=dist+min_dist_to_a_frame
                    count=count+1
                # Dist now contains 
            #Avg min dist
            pred[d]=(dist/count)
            #print(str(d)+":"+str(pred[d]))
        predic=np.argmin(pred)
        if(predic==digit):
            accuracy=accuracy+1
            accuracy_perd[digit]=accuracy_perd[digit]+1
        print("Predicted: "+ str(predic)+" for expected digit: "+str(digit))
print("Net Accuracy")
print(accuracy/640)
print("Accuracy Per Digit")
print(accuracy_perd/64)