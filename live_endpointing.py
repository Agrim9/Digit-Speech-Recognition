#hBased on Paper by Rabiner and Sambur http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6778857
from scipy import signal
from scipy.io.wavfile import write,read
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import convolve
from scipy.signal import hamming
from scipy.ndimage.interpolation import shift
from scipy.linalg import toeplitz,inv
from scipy.fftpack import fft,fftshift,ifft


def live_endpointer(inp,Fs):
    Ts=1/Fs
    #plt.title("Raw Input")
    #plt.plot(inp,'blue')
    
    #------------------------------------------------------------------------------
    #Compute Avg magnitude (Short Time), with window size N/20 (0-9 spoken twice)
    W=1000
    pad_zeros=np.zeros(W)
    zp_inp_t=np.append(pad_zeros,inp)
    zp_inp=np.append(zp_inp_t,pad_zeros)
    st_avg=np.array([np.sum(abs(zp_inp[i-W:i+W])) for i in range(W,zp_inp.size-W)])
    print(st_avg.size)
    print(inp.size)
    #plt.plot(st_avg,'red')

    #Compute IZCT assuming 0.2s and 0.2 silence in end
    num_samp_in_sil=int(0.2*Fs)
    sil_inp_st=st_avg[0:num_samp_in_sil]
    sil_inp_end=st_avg[st_avg.size-num_samp_in_sil:st_avg.size]
    sil_inp=np.append(sil_inp_st,sil_inp_end)
    #Compute ZCR
    
    #Compute Peak Energy Imax and Silence Energy Imin
    Imx=np.amax(st_avg)
    Imn=np.amax(sil_inp)
    #Compute ITL and ITU
    I1=0.3*(Imx-Imn)+Imn
    I2=50*Imn
    ITL=Imx/10
    #ITL=Imx/10
    print("ITL is "+str(ITL))
    plt.plot(st_avg)
    ITU=1.2*ITL
    
    #Search Fwd
    start_index=num_samp_in_sil+1
    end_index=inp.size-1
    zero_st,zero_fi=search(st_avg,start_index,end_index,ITU,ITL)
    

    #write('test.wav', 8000, np.array(inp[int(zero_st/2):int(zero_fi/2)]).astype(np.dtype('i2')))
    return inp[zero_st:zero_fi],zero_st,zero_fi

def search(st_avg,start_index,end_index,ITU,ITL):
    N1=0
    N2=0
    for i in range(start_index,end_index):
        flag=0
        if(st_avg[i]>ITL):
            print("Here")
            N1=i
            break
    for i in range(end_index,start_index,-1):
        flag=0
        if(st_avg[i]>ITL):
            N2=i
            break
    return int(N1/2),int(N2/2)

