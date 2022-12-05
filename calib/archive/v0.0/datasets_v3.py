import os
from calib.const import EPS
import numpy as np 
import cv2 as cv
import sys
import os

class BoxDataLoader():

    def __init__(self, dataset_dir,
                       seq_name           = 'MOT20-02',
                       window             = 60,      # size of time window
                       stride             = 30,      # stride of time window
                       height_dif_thresh  = 2,       # threshold for height difference (image plane)
                       front_ratio_thresh = 0.8,     # threshold for the ratio of being as front tracks
                       fps                = 30):

        self.track_dir          = f'{dataset_dir}/{seq_name}'
        self.tracks             = np.load(self.track_dir + '/masked_tracks.npy') # track data: num_tracks x num_frames x 4
        self.track_ids          = np.load(self.track_dir + '/track_ids.npy')
        self.front_masks        = np.load(self.track_dir + '/front_mask.npy')
        self.window             = window                    # time window size
        self.stride             = stride                    # stride for sliding window
        self.loc                = 'center'
        self.front_ratio_thresh = front_ratio_thresh
        self.height_dif_thresh  = height_dif_thresh
        self.fps                = fps

        self.box_pairs          = self.gen_box_pairs()

    def get_ids(self):
        return self.track_ids
    
    def get_all_pairs(self):
        return self.box_pairs

    def gen_box_pairs(self):
        seq_len = self.tracks.shape[1]
        num_batchs = (seq_len - self.window) // self.stride
        batchs = []
        for i in range(num_batchs):

            # track boxes in a time window
            track_boxes = self.tracks[:, i * self.stride:\
                i * self.stride + self.window, :]  # N x W x 4
            
            # a mask of front tracks
            front_mask = self.front_masks[:, i * self.stride:\
                i * self.stride + self.window]  # N x W x 4
            
            # select continous and high-front-ratio tracks
            ind_mask = get_ind_mask(track_boxes[:,:,-1], front_mask) 
            masked_track_boxes = track_boxes[ind_mask, :, :]
            box_pairs = masked_track_boxes[:, [0, - 1], :]
            
            # reject pairs of boxes that height difference is small
            box_pairs = select_pairs_by_height_dif(box_pairs, height_dif_thresh = self.height_dif_thresh)

            batchs.append(box_pairs)
        
        self.batchs = batchs

        return np.concatenate(self.batchs, axis = 0)

    def get_batch_counts(self):
        return len(self.batchs)
    
    def get_batch(self, batch_index):
        return self.batchs[batch_index]

# select continous and front tracks in a time window
def get_ind_mask(mask, front_mask, front_ratio_thresh = 0.8):
    ind_mask = np.ones((mask.shape[0],)) == 1
    box_mask     =  mask > 0                 # N x W
    tracklet_len = box_mask.sum(axis = -1)   # N
    front_times  = front_mask.sum(axis = -1) 
    front_ratio  = front_times / (tracklet_len + EPS)

    ind_mask = np.logical_and(tracklet_len == mask.shape[1],    # continous tracklet
                              front_ratio > front_ratio_thresh, # front track ratio should be high
                              )

    return ind_mask

# box_pairs: N x 2 x 4
# new_box_pairs: M x 2 x 4
def select_pairs_by_height_dif(box_pairs, height_dif_thresh = 3):
    h1    = box_pairs[:, 0, -1]
    h2    = box_pairs[:, 1, -1]
    inds  = np.where(np.abs(h1 - h2) > height_dif_thresh)[0]
    return box_pairs[inds, :, :]