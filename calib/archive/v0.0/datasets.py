import os
import numpy as np 
import cv2 as cv
import sys
import os

class BoxDataLoader():

    def __init__(self, dataset_dir,
                       seq_name    = 'MOT20-02',
                       window      = 50,      # size of time window
                       stride      = 10,      # stride of time window
                       interval    = 10,      # sampling interval
                       h_dif_thres = 3,    # threshold for height difference (image)
                       trj_thres   = 0.9,  # threshold for the inter-product between 
                                           # a trajectory and the projection
                       fps         = 20):

        self.track_dir      = f'{dataset_dir}/{seq_name}'
        self.tracks         = np.load(self.track_dir + '/masked_tracks.npy') # track data: num_tracks x num_frames x 4
        self.track_ids      = np.load(self.track_dir + '/track_ids.npy')
        self.window         = window     # time window size
        self.stride         = stride     # stride for sliding window
        self.batchs         = None       # batches for calibration
        self.interval       = interval   # subsmpling interval
        self.ids_to_indices = {}
        self.loc            = 'center'

        for i in range(self.track_ids.shape[0]):
            self.ids_to_indices[self.track_ids[i]] = i

        self.ind_mask = np.ones((len(self.track_ids),))
    
        self.h_dif_thres   = h_dif_thres
        self.fps           = fps
        self.trj_thres     = trj_thres

        self.gen_box_pairs()
        self.get_height_box_pairs()
        self.comb_all_height_pairs()

    def get_ids(self):
        return self.track_ids

    def gen_box_pairs(self):
        seq_len = self.tracks.shape[1]
        num_batchs = (seq_len - self.window) // self.stride
        batchs = []
        for i in range(num_batchs):
            track_boxes = self.tracks[:, i * self.stride:\
                i * self.stride + self.window, :]  # N x W x 4
            ind_mask = get_ind_mask(track_boxes[:,:,-1], self.ind_mask)
            masked_track_boxes = track_boxes[ind_mask, :, :]
            box_pairs = gen_pairs(masked_track_boxes, self.loc, self.trj_thres, self.fps)
            batchs.append(box_pairs)
        self.batchs = batchs

    def get_height_box_pairs(self):
        h_batchs = []
        for i in range(len(self.batchs)):
            if len(self.batchs[i]) == 0: 
                h_batchs.append(np.array([]))
                continue
            # print(len(self.batchs[i]))
            selected_pairs = select_pairs_by_height_dif(self.batchs[i], 
                                                        self.h_dif_thres)
            h_batchs.append(selected_pairs)
        self.h_batchs = h_batchs

    def comb_all_height_pairs(self):
        all_h_pairs = [batch.reshape((-1, 4, 4))\
                       for batch in self.h_batchs if len(batch) > 0]
        self.all_h_pairs = np.concatenate(all_h_pairs, axis = 0)
    
    def get_all_hpairs(self):
        return self.all_h_pairs

    def get_batch_counts(self):
        return len(self.batchs)
    
    def get_batch(self, batch_index):
        return self.batchs[batch_index]

# box_pairs: N x 4 x 4
# new_box_pairs: M x 4 x 4
def get_box_pts(boxes, loc = 'center'):
    if loc == 'center':
        box_pts = boxes[:, :, :2] + boxes[:, :, 2:] / 2
    elif loc == 'bottom':
        box_pts = boxes[:, :, :2].copy()
        box_pts[:, :, 0] += boxes[:, :, 2] / 2 # x + w/2
        box_pts[:, :, 1] += boxes[:, :, 3]     # y + h
    return box_pts

# box points in a window
def gen_pairs(box_pts, loc, thres, fps = 20):
    if len(box_pts) == 0: return np.array([])
    pair1 = box_pts[:, :fps, :]
    # pair2 = box_pts[:, int(fps * 3/2):int(fps * 3/2)+fps, :]
    pair2 = box_pts[:, int(fps * 3/2):int(fps * 3/2)+fps, :]

    cond1 = trajectory_check(pair1, loc, thres)
    cond2 = trajectory_check(pair2, loc, thres)
    cond  = np.logical_and(cond1, cond2).flatten()

    inds = np.where(cond)[0]
    used_pair1 = pair1[inds, :, :]
    used_pair2 = pair2[inds, :, :]

    valid_pairs = np.concatenate([used_pair1[:, [0, -1], :],\
                                  used_pair2[:, [0, -1], :]], axis = 1)
    return valid_pairs

# N x 2, N x 2
def inner_prod(major_vs, minor_vs):
    prods   = np.sum(major_vs * minor_vs, axis = -1)
    cosines = prods / np.linalg.norm(major_vs, axis = -1)\
              / np.linalg.norm(minor_vs, axis = -1)
    return cosines

# N x 2
def trajectory_check(box_trj, loc, thres = 0.9):
    box_trj  = get_box_pts(box_trj, loc)
    major_vs = box_trj[:, -1, :] - box_trj[:, 0, :]
    cosine_sum = 0
    for i in range(1, box_trj.shape[1]):
        minor_vs    = box_trj[:, i, :] - box_trj[:, i - 1, :]
        cosines     = inner_prod(major_vs, minor_vs)  # N x 2, N x 2
        cosine_sum += cosines
    cosine_mean = cosine_sum / (box_trj.shape[1] - 1)
    return cosine_mean > thres

# check if a track is valid in a time window
def get_ind_mask(mask, ori_ind_mask):
    ind_mask = ori_ind_mask.copy() == 1
    for i in range(mask.shape[-1]):
        ind_mask = np.logical_and(ind_mask, mask[:, i])
    return ind_mask

# downsample the boxes
def downsample(boxes, sample_interval):
    return boxes[::sample_interval, :]

# box_pairs: N x 4 x 4
# new_box_pairs: M x 4 x 4
def select_pairs_by_height_dif(box_pairs, h_dif_thres = 3):
    h1    = box_pairs[:, 1, -1]
    h2    = box_pairs[:, 3, -1]
    inds  = np.where(np.abs(h1 - h2) > h_dif_thres)[0]
    return box_pairs[inds, :, :]
