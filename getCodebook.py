# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:47:19 2017

@author: suray
"""

import numpy as np
import cv2
import pickle

from cyvlfeat import sift as cysift
from scipy.cluster.vq import kmeans

imgs= list();
for j in range(1000):
    if j%2 == 0:
        filename = str(j) + ".jpg"
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(gray)
        print('image ', j)
    

feature = list();   
for i in range(500):
    gray = imgs[i]
    (fv, des) = cysift.sift(gray, peak_thresh=3, edge_thresh=7, compute_descriptor=True)
    feature.append((fv, des))


des = [item[1] for item in feature]
alldes = np.vstack(des)

k = 128
alldes = np.float32(alldes)      # convert to float, required by kmeans and vq functions
codebook, distortion = kmeans(alldes, k)


pickle.dump( codebook, open( "codebook.pkl", "wb" ) )
print('Features pickled!')