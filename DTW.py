import numpy as np
from numpy.linalg import norm
from dtw import dtw
accuracy=0
accuracy_perd=np.zeros(10)

for speak in range(1,65):
    print("Speaker "+ str(int((speak-1)/4)+1)+" Uterrence: "+ str(int(speak-1)%4+1))
    for digit in range(10):
        # Read from File the MFCC of the digit under analysis of the speaker
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
                # Read from File the MFCC of the digit under analysis of the other speakers
                filename="./Extracted_Feats/"+str(d)+"/"+str(i)+".npy"
                tarr=np.load(filename)
                # DTW distance between the two MFCCs (zr == testing), (tarr= reference)
                dtw_dist = dtw(zr, tarr, dist=lambda x, y: norm(x - y, ord=1))
                dist=dist+dtw_dist[0]
                count=count+1

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