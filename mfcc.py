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

def get_mel_filters(N,num_filts,Fs,freq_down=0,freq_up=4000):
    up_melcoef=1125*np.log(1+freq_up/700)
    down_melcoef=1125*np.log(1+freq_down/700)
    filtbank_mel=np.linspace(down_melcoef,up_melcoef,num_filts+2)
    filtbank_hz=700*(np.exp(filtbank_mel/1125)-1)
    #print (filtbank_hz)
    filtbank_bins=np.floor((N+1)*filtbank_hz/Fs)
    print(filtbank_bins)
    #print (filtbank_bins)
    #Computing filters now
    filters=np.zeros(shape=(num_filts,N))
    for i in range(1,num_filts//2+1):
        fp=int(filtbank_bins[i-1])
        sp=int(filtbank_bins[i])
        tp=int(filtbank_bins[i+1])
        filters[i-1]=np.hstack((np.zeros(fp),np.linspace(0,1,sp-fp+1),np.linspace(1,0,tp-sp+1)[1:],np.zeros(N-tp-1))).ravel()
        #plt.plot(filters[i-1])
    return filters

def feat_ext(digit,num_wav,Fs):
    
    # Parameters
    N=512
    num_filts=20
    mel_filters=get_mel_filters(N,num_filts,Fs)
    t_analysis=0.01
    hop_samp=int(Fs*t_analysis)
    num_samp=hop_samp
    
    for samp_under_analysis in range(3,7):

        filename="./Processed_Data/"+str(digit)+"/"+str(samp_under_analysis)+".wav"
        print(filename+ " Read")
        read_wav = read(filename)
        inp_wo_emph=np.array(read_wav[1],dtype='float64')
        inp = np.zeros(inp_wo_emph.size)
        a=0.97
        for k in range (inp_wo_emph.size):
            if(k==0):
                inp[k]=inp_wo_emph[k]
            else:
                inp[k]=inp_wo_emph[k] - a*inp_wo_emph[k-1];

        
        #plt.rcParams["figure.figsize"] = (18,10)
        #plt.plot(inp,'blue')
        num_feat=int((inp.size-num_samp)/hop_samp)
        num_coef=13
        mfcc=np.zeros(shape=(num_feat,num_coef))
        st_ind=0
        print (inp.size)
        for i in range(num_feat):
    
            end_ind=st_ind+num_samp            
            #print ("Frame number: "+str(i)+" st index "+ str(st_ind)+" end index "+ str(end_ind))
            frame=inp[st_ind:end_ind]
            #---------------------------------------------------------
            #Windowed DFT
            zero_arr=np.zeros(N-num_samp)
            zero_pd=np.append(frame,zero_arr)
            hamm_w=np.append(hamming(num_samp),np.zeros((N-num_samp)))
            s_n=zero_pd*hamm_w ## x[n]=s[n]w[n]
            S_k=fft(s_n)
            #P_k=(np.abs(S_k[0:257])*np.abs(S_k[0:257]))/num_samp
            P_k=np.abs(S_k)
            #---------------------------------------------------------
            #Computing MFCC
            energy_fbank=[20*np.log10(np.sum(P_k*mel_filters[i])) for i in range(num_filts)]
            ifft_samp=ifft(energy_fbank)
            #---------------------------------------------------------
            # DCT
            #orig_mfcc=dct(energy_fbank)
            orig_mfcc=ifft_samp
            #if(i>=2 and i<(num_feat-2)):
            #    delta_mfcc=(mfcc[i+1]-mfcc[i-1]+2*mfcc[i+2]-2*mfcc[i-2])/10
            #else:
            #    delta_mfcc=np.zeros(13)
            
            #mfcc[i]=np.append(orig_mfcc,delta_mfcc[0:7])
            mfcc[i]=orig_mfcc
            #print (mfcc[i])
            st_ind=st_ind+hop_samp
        filename="./Extracted_Feats/"+str(digit)+"/"+str(samp_under_analysis)
        #print(filename+" Saved")
        #np.save(filename,mfcc)
  
feat_ext(0,2,8000)
feat_ext(3,2,8000)
            