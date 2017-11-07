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

accuracy4=0
accuracy8=0
accuracy_pred4=np.zeros(10)
accuracy_pred8=np.zeros(10)

for speak in range(1,65):
    print("Speaker "+ str(int((speak-1)/4)+1)+" Uterrence: "+ str(int(speak-1)%4+1))
    for digit in range(10):
        filename="./Extracted_Feats_ib/"+str(digit)+"/"+str(speak)+".npy"
        zr=np.load(filename)
        #print(exc_samp)
        pred4=np.zeros(10)
        pred8=np.zeros(10)
        for d in range(10):
            dist4=0
            dist8=0
            count=1
            filename="./VQ_codebooks/Speaker "+str(int((speak-1)/4)+1)+"/"+str(d)+"/k4.npy"
            k4=np.load(filename)
            filename="./VQ_codebooks/Speaker "+str(int((speak-1)/4)+1)+"/"+str(d)+"/k8.npy"
            k8=np.load(filename)
            for j in range(zr.shape[0]):
                    # Find Closest Matching Frame
                    # Instead of adding distancesto all centroids, add distance to one centroid min
                    begin=True
                    min_dist_to_a_centroid4=0
                    for centroid in k4:
                        obt_dist=np.sum(np.abs(zr[j]-centroid))
                        if(begin):
                            min_dist_to_a_centroid4=obt_dist
                            begin=False
                        else:
                            if(obt_dist<min_dist_to_a_centroid4):
                                min_dist_to_a_centroid4=obt_dist
                    begin=True
                    min_dist_to_a_centroid8=0
                    for centroid in k8:
                        obt_dist=np.sum(np.abs(zr[j]-centroid))
                        if(begin):
                            min_dist_to_a_centroid8=obt_dist
                            begin=False
                        else:
                            if(obt_dist<min_dist_to_a_centroid8):
                                min_dist_to_a_centroid8=obt_dist
                    dist4=dist4+min_dist_to_a_centroid4
                    dist8=dist8+min_dist_to_a_centroid8
                    count=count+1
                # Dist now contains 
            #Avg min dist
            pred4[d]=(dist4/count)
            pred8[d]=(dist8/count)
            #print(str(d)+":"+str(pred[d]))
        predic4=np.argmin(pred4)
        if(predic4==digit):
            accuracy4=accuracy4+1
            accuracy_pred4[digit]=accuracy_pred4[digit]+1
        predic8=np.argmin(pred8)
        if(predic8==digit):
            accuracy8=accuracy8+1
            accuracy_pred8[digit]=accuracy_pred8[digit]+1

        print("Predicted: "+ str(predic4)+ " for expected digit: "+str(digit) + " k4")
        print("Predicted: "+str(predic8)+" for expected digit: "+str(digit) + " k8")

print("Net Accuracy for k=4")
print(accuracy4/640)
print("Accuracy Per Digit for k=4")
print(accuracy_pred4/64)
print("Net Accuracy for k=8")
print(accuracy8/640)
print("Accuracy Per Digit for k=8")
print(accuracy_pred8/64)