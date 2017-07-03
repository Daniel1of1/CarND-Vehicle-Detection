import cv2
from features import *
from scipy.ndimage.measurements import label
import numpy as np

class VehicleDetector:

    classifier = None
    
    def __init__(self,classifier):
        self.classifier = classifier
    
    def detect_vehicles(self,img):
        
        hot_windows = self.detections(img)
        
        heat = heatmap(img.shape[:2],hot_windows)
    
        thresh_heat = apply_threshold(heat,5)
    
        bboxes = bounding_boxes(thresh_heat)

        return bboxes
    
    def detections(self, img):
        ystart = 400
        ystop = 656
        scale = 1.5
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = 16
        hist_bins = 32

        hot_windows = []
        hot_windows += self.positive_detections(img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        hot_windows += self.positive_detections(img, 400, 528, 1, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        hot_windows += self.positive_detections(img, 400, 464, 0.75, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        hot_windows += self.positive_detections(img, 400, 464, 0.5, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        return hot_windows


    def positive_detections(self,
                            img,
                            ystart,
                            ystop,
                            scale,
                            orient,
                            pix_per_cell,
                            cell_per_block,
                            spatial_size,
                            hist_bins):
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = hog_features(ch1)
        hog2 = hog_features(ch2)
        hog3 = hog_features(ch3)


        hot_windows = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features1 = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=(spatial_size,spatial_size))
                hist_features = color_hist_vec(subimg, nbins=hist_bins)

                feature_vec = np.hstack((hog_features1, hist_features, spatial_features)).reshape(1, -1)
                
                test_prediction = self.classifier.predict(feature_vec)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    hot_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

        return hot_windows


def heatmap(size, boxes):
    heat = np.zeros(size).astype(np.float)    
    for box in boxes:
        if len(box) == 2:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
    return heat

        
def apply_threshold(heatmap, threshold):
    new = heatmap.copy()
    # Zero out pixels below the threshold
    new[new <= threshold] = 0
    # Return thresholded map
    return new


    
def bounding_boxes(heatmap):
    bboxes = []
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        
    return bboxes