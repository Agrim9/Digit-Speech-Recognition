import numpy as np
from dtw import dtw
from numpy.linalg import norm

accuracy=0
accuracy_perd=np.zeros(10)
for speak in range(1,65):
    print("Speaker "+ str(int((speak-1)/4)+1)+" Uterrence: "+ str(int(speak-1)%4+1))
    for digit in range(10):
        filename="./Extracted_Feats_ib/"+str(digit)+"/"+str(speak)+".npy"
        zr=np.load(filename)
        forbidden=[int(speak/4)*4+1,int(speak/4)*4+2,int(speak/4)*4+3,int(speak/4)*4+4]
        exc_samp=[i for i in range(1,65) if i not in forbidden]
        #print(exc_samp)
        pred=np.zeros(10)
        for d in range(10):
            dist=0
            count=1
            for i in exc_samp:
                filename="./Extracted_Feats_ib/"+str(d)+"/"+str(i)+".npy"
                tarr=np.load(filename)
                dtw_dist = dtw(zr, tarr, dist=lambda x, y: norm(x - y, ord=1))
                #print(dtw_dist[0])

                dist=dist+dtw_dist[0]
                count=count+1
                '''
                # Find all closest distances to the frames
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
                '''
            #Avg min dist
            pred[d]=(dist/count)
            #print(str(d)+":"+str(pred[d]))
        predic=np.argmin(pred)
        if(predic==digit):
            accuracy=accuracy+1
            accuracy_perd[digit]=accuracy_perd[digit]+1
        print("Predicted: "+ str(predic)+" for expected digit: "+str(digit))
print("Net Accuracy")
print(accuracy/64)
print("Accuracy Per Digit")
print(accuracy_perd/4)