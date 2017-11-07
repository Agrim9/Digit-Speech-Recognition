from scipy import signal
from scipy.io.wavfile import write,read
import math
import numpy as np
from sklearn.cluster import KMeans

for speaker in range(1,17):
    print("VQ Codebooks for Speaker "+ str(speaker))
    for digit in range(10):
        forbidden=[(speaker-1)*4+1,(speaker-1)*4+2,(speaker-1)*4+3,(speaker-1)*4+4]
        print(forbidden)
        exc_samp=[i for i in range(1,65) if i not in forbidden]
        list_of_frames=[]
        # Concat all feature vectors for the particular digit
        for i in exc_samp:
            filename="./Extracted_Feats/"+str(digit)+"/"+str(i)+".npy"
            tarr=np.load(filename)
            for k in range(tarr.shape[0]):
                list_of_frames.append(tarr[k])
        arr_of_frames=np.array(list_of_frames)
        #Do k means appropriately
        kmeans4 = KMeans(n_clusters=4).fit(arr_of_frames)
        kmeans8 = KMeans(n_clusters=8).fit(arr_of_frames)
        print("Kmeans 4")
        print(kmeans4.cluster_centers_)
        print("Kmeans 8")
        print(kmeans8.cluster_centers_)
        filename="./VQ_codebooks/Speaker "+str(speaker)+"/"+str(digit)+"/k4"
        np.save(filename,kmeans4.cluster_centers_)
        filename="./VQ_codebooks/Speaker "+str(speaker)+"/"+str(digit)+"/k8"
        np.save(filename,kmeans8.cluster_centers_)

