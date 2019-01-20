from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.interpolate import interp1d
import cv2
import numpy as np
import os
from cyvlfeat import sift as cysift
from scipy.cluster.vq import vq
import pickle


dbpath = 'C:\\Users\\suray\\Documents\\VIP\\as2\\fooddb' 

fvbaseline=[]
fvsift=[]
codebook = pickle.load(open("codebook.pkl", "rb"))

for i in range(1000):
    #read image
    img = cv2.imread( os.path.join(dbpath, str(i) + ".jpg") )  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #fv for rgb histogram method
    rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
    ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
    bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
    features = np.concatenate((rhist, ghist, bhist))
    fvbaseline.append(features)
    
    #fv for sift + rgb histogram method
    f, descriptor = cysift.sift(gray, peak_thresh=1, edge_thresh=10, compute_descriptor=True)    
    code, distortion = vq(descriptor, codebook)    
    hist = np.histogram(code, 128, normed=True)
    feat=hist[0]
    srhist, rbins = np.histogram(img[:,:,0], 16, normed=True)
    sghist, gbins = np.histogram(img[:,:,1], 16, normed=True)
    sbhist, bbins = np.histogram(img[:,:,2], 16, normed=True)
    features2 = np.concatenate((feat, srhist, sghist, sbhist))
    fvsift.append(features2)
                
    
#converting fvbaseline and fvsift to array
temparr = np.array(fvbaseline)
fvbaseline = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
del temparr

temparr = np.array(fvsift)
fvsift = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
del temparr
          
D = distance.squareform(distance.pdist(fvbaseline, 'braycurtis') )
avg_prec = np.zeros(1000)
recall = np.zeros(1000)
nPerCategory = 100       
nCategory = 10            
nRetrieved = 100  

D2 = distance.squareform(distance.pdist(fvsift, 'braycurtis') )
avg_prec2 = np.zeros(1000)
recall2 = np.zeros(1000)


arr=np.zeros((1000,2))
arr2=np.zeros((1000,2))

for c in range(nCategory): 
    for i in range(nPerCategory):
        idx = (c*nPerCategory) + i;
        
        nearest_idx = np.argsort(D[idx, :]);

        retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/nPerCategory));

        hits = (retrievedCats == np.floor(idx/nPerCategory))

        if np.sum(hits) != 0:
            avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(nRetrieved)+1)) / np.sum(hits)
        else:
            avg_prec[idx] = 0.0
      
        recall[idx] = np.sum(hits) / nPerCategory

        arr[idx]=(recall[idx], avg_prec[idx])
        
for d in range(nCategory): 
    for j in range(nPerCategory):
        idx = (c*nPerCategory) + j;
        
        nearest_idx2 = np.argsort(D2[idx, :]);

        retrievedCats2 = np.uint8(np.floor((nearest_idx2[1:nRetrieved+1])/nPerCategory));
 
        hits2 = (retrievedCats2 == np.floor(idx/nPerCategory))

        if np.sum(hits2) != 0:
            avg_prec2[idx] = np.sum(hits2*np.cumsum(hits2)/(np.arange(nRetrieved)+1)) / np.sum(hits2)
        else:
            avg_prec2[idx] = 0.0
      
        recall2[idx] = np.sum(hits2) / nPerCategory

        arr2[idx] = (recall2[idx], avg_prec2[idx])

arr = arr[np.argsort(arr[:,0])]
arr2 = arr2[np.argsort(arr2[:,0])]
            
plt.plot(arr[:,0], arr[:,1], 'b--', arr2[:,0], arr2[:,1], 'g')
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Precision-Recall Curve")
plt.legend(('RGB', 'SIFT + RGB' ), loc='upper left')
plt.xlim(0,0.5)
plt.ylim(0,0.8)
#plt.plot(arr[0], sortrecall[1])
plt.show()