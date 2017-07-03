import cv2
import numpy as np
from skimage.feature import hog


def color_hist_vec(img, nbins=32):
    # Compute the histogram of the YcrCb channels separately
    yhist = np.histogram(img[:,:,0], bins=32)
    crhist = np.histogram(img[:,:,1], bins=32)
    cbhist = np.histogram(img[:,:,2], bins=32)
    # Generating bin centers
    bin_edges = yhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((yhist[0], crhist[0], cbhist[0]))
    # Return feature vector
    return hist_features


def bin_spatial(img, size=(16, 16)):
    feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

def hog_features(img):    
    return hog(img, 9, pixels_per_cell = (8,8), cells_per_block = (2,2), visualise=False, feature_vector=False)


def full_feature(img):
    feature_img = img
    feature_img = cv2.cvtColor(feature_img,cv2.COLOR_RGB2YCrCb)
    
    hogs = []
    for i in range(feature_img.shape[2]):
        channel = feature_img[:,:,i]
        channel_hog_feature = hog_features(channel).ravel()
        hogs.append(channel_hog_feature)
    
    hogs = np.hstack(hogs)
    color_hist_feature = color_hist_vec(feature_img)
    
    spatial = bin_spatial(feature_img)
        
    features = np.hstack([hogs,color_hist_feature,spatial])
    
    return features
