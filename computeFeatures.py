"""
computeFeatures.py

YOUR WORKING FUNCTION for computing features

"""
import numpy as np
import cv2
import pickle
from cyvlfeat import sift as cysift
from scipy.cluster.vq import vq

# you are allowed to import other Python packages above
##########################
def computeFeatures(img):
    # Inputs
    # img: 3-D numpy array of an RGB color image
    #
    # Output
    # featvect: A D-dimensional vector of the input image 'img'
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    codebook = pickle.load(open("codebook.pkl", "rb"))       
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f, descriptor = cysift.sift(gray, peak_thresh=1, edge_thresh=10, compute_descriptor=True)
    
    code, distortion = vq(descriptor, codebook)    
    hist = np.histogram(code, 128, normed=True)
    feat=hist[0]
    
    rhist, rbins = np.histogram(img[:,:,0], 16, normed=True)
    ghist, gbins = np.histogram(img[:,:,1], 16, normed=True)
    bhist, bbins = np.histogram(img[:,:,2], 16, normed=True)
    featvect = np.concatenate((feat, rhist, ghist, bhist))
     
    # END OF YOUR CODE
    #########################################################################
    return featvect
