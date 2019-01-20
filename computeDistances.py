"""
computeDistances.py

YOUR WORKING FUNCTION for computing pairwise distances between features

"""
from scipy.spatial import distance
import numpy as np
from scipy.cluster.vq import kmeans, vq

# you are allowed to import other Python packages above
##########################
def computeDistances(fv):
    # Inputs
    # fv: A N-by-D array containing D-dimensional feature vector of 
    #     N number of data (images)
    # 
    # Output
    # D: N-by-N square matrix containing the pairwise distances between
    #    all samples, i.e. the first row shows the distance
    #    between the first sample and all other samples 
    #    (columns)
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    D = distance.squareform(distance.pdist(fv, 'braycurtis') )
    
        
        
    # END OF YOUR CODE
    #########################################################################
    return D