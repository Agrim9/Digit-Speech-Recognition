from scipy import signal
from scipy.io.wavfile import write,read
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import convolve
from scipy.signal import hamming
from scipy.ndimage.interpolation import shift
from scipy.linalg import toeplitz,inv
from scipy.fftpack import fft,fftshift,ifft,dct
from python_speech_features import *

def get_mel_filters(N,num_filts,Fs,freq_down=0,freq_up=4000):
    up_melcoef=1125*np.log(1+freq_up/700)
    down_melcoef=1125*np.log(1+freq_down/700)
    filtbank_mel=np.linspace(down_melcoef,up_melcoef,num_filts+2)
    filtbank_hz=700*(np.exp(filtbank_mel/1125)-1)
    #print (filtbank_hz)
    filtbank_bins=np.floor((2*N-2)*filtbank_hz/Fs)+1
    #print(filtbank_hz)
    print (filtbank_bins)
    #Computing filters now
    filters=np.zeros(shape=(num_filts,N))
    freq=np.arange(N)*(Fs/(2*N-2))
    for i in range(1,num_filts+1):
        fp=int(filtbank_bins[i-1])
        sp=int(filtbank_bins[i])
        tp=int(filtbank_bins[i+1])
        filters[i-1]=np.hstack((np.zeros(fp),np.linspace(0,1,sp-fp+1),np.linspace(1,0,tp-sp+1)[1:],np.zeros(N-tp-1))).ravel()
        #plt.plot(freq,filters[i-1])
    return filters

def feat_ext(digit,num_wav,Fs,use_inbuilt=0):
    
    # Parameters
    N=512
    num_filts=26
    mel_filters=get_mel_filters(N//2+1,num_filts,Fs)
    t_analysis=0.01
    hop_samp=int(Fs*t_analysis)
    num_samp=hop_samp
    plt.show()
    for samp_under_analysis in range(1,num_wav+1):

        filename="./Processed_Data/"+str(digit)+"/"+str(samp_under_analysis)+".wav"
        print(filename+ " Read")
        read_wav = read(filename)
        inp_wo_emph=np.array(read_wav[1],dtype='float64')
        #inp=np.array(read_wav[1],dtype='float64')
        if(use_inbuilt==1):
            mfcc_feat=mfcc(inp_wo_emph,samplerate=8000,winlen=0.01,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
        
        else:
            a=0.97
            # Pre Emphasis
            # Doing Pre Emphasis at this stage since I'd forgotten to do it in end pointing stage
            inp = np.append(inp_wo_emph[0], inp_wo_emph[1:]-a*inp_wo_emph[:-1])
            
            #plt.rcParams["figure.figsize"] = (18,10)
            #plt.plot(inp,'blue')
            
            num_feat=int((inp.size-num_samp)/hop_samp)
            num_coef=13
            mfcc_feat=np.zeros(shape=(num_feat,num_coef))
            st_ind=0
            #print (inp.size)
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
                #P_k=np.square(np.abs(S_k)[0:257])/num_samp
                P_k=np.abs(S_k[0:257])
                
                #---------------------------------------------------------
                #Computing MFCC
                epsilon=1e-8
                energy_fbank=[20*np.log10(np.sum(P_k*mel_filters[i])+epsilon) for i in range(num_filts)]
                ifft_samp=ifft(energy_fbank)
                #---------------------------------------------------------
                # Delta Delta Computation
                orig_mfcc=ifft_samp[0:13]
                #if(i>=2 and i<(num_feat-2)):
                #    delta_mfcc[i]=(mfcc[i+1]-mfcc[i-1]+2*mfcc[i+2]-2*mfcc[i-2])/10
                #else:
                #    delta_mfcc=np.zeros(13)
                
                #if(i>=4 and i<(num_feat-4)):
                #    dealt_delta_mfcc=(mfcc[i+1]-mfcc[i-1]+2*mfcc[i+2]-2*mfcc[i-2])/10
                    
                #mfcc[i]=np.append(orig_mfcc,delta_mfcc[0:13])
                mfcc_feat[i]=orig_mfcc
                #print (mfcc[i])
                st_ind=st_ind+hop_samp
        #print(mfcc_feat[num_feat-1])
        filename="./Extracted_Feats/"+str(digit)+"/"+str(samp_under_analysis)
        print(filename+" Saved")
        #np.save(filename,mfcc_feat)
  
feat_ext(0,64,8000)
feat_ext(1,64,8000)
feat_ext(2,64,8000)
feat_ext(3,64,8000)
feat_ext(4,64,8000)
feat_ext(5,64,8000)
feat_ext(6,64,8000)
feat_ext(7,64,8000)
feat_ext(8,64,8000)
feat_ext(9,64,8000)
            