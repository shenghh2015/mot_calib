import os
import numpy as np 
import cv2 as cv
import sys
import os

class BoxDataLoader():

    def __init__(self, track_file         = '.',
                       img_size           = (1920, 1080),
                       window             = 60,      # size of time window
                       stride             = 30,      # stride of time window
                       height_dif_thresh  = 2,       # threshold for height difference (image plane)
                       front_ratio_thresh = 0.8,     # threshold for the ratio of being as front tracks
                       fps                = 30,
                       use_clip           = True,
                       use_front          = True):

        # self.track_dir          = f'{dataset_dir}/{seq_name}'
        # self.tracks             = np.load(self.track_dir + '/masked_tracks.npy') # track data: num_tracks x num_frames x 4
        # self.track_ids          = np.load(self.track_dir + '/track_ids.npy')
        # self.front_masks        = np.load(self.track_dir + '/front_mask.npy')
        self.window             = window                    # time window size
        self.stride             = stride                    # stride for sliding window
        self.front_ratio_thresh = front_ratio_thresh
        self.height_dif_thresh  = height_dif_thresh
        self.fps                = fps
        self.use_clip           = use_clip
        self.use_front          = use_front

        self.h_img              = img_size[1]

        # self.track_file         = self.track_dir + '/masked_tracks.npy'
        self.track_file         = track_file

        self.raw_tracks, _      = self.load_raw_tracks()

    def load_raw_tracks(self):
        # load tracks from txt file
        base_name = os.path.basename(self.track_file)

        if base_name.endswith('gt.txt'):
            track_boxes, track_ids = load_tracks_from_gt(self.track_file)
        else:
            track_boxes, track_ids = load_tracks_from_tracking(self.track_file)

        return track_boxes, track_ids

# select continous and front tracks in a time window
def get_ind_mask(mask, front_mask, front_ratio_thresh = 0.8):
    ind_mask     = np.ones((mask.shape[0],)) == 1
    box_mask     =  mask > 0                 # N x W
    tracklet_len = box_mask.sum(axis = -1)   # N
    front_times  = front_mask.sum(axis = -1) 
    front_ratio  = front_times / (tracklet_len + EPS)

    # ind_mask = np.logical_and(tracklet_len == mask.shape[1],    # continous tracklet
    #                           front_ratio > front_ratio_thresh, # front track ratio should be high
    #                           )

    ind_mask = front_ratio > front_ratio_thresh
    ind_mask = np.logical_and(np.logical_and(box_mask[:, 0], box_mask[:, -1]), ind_mask)

    return ind_mask

# box_pairs: N x 2 x 4
# new_box_pairs: M x 2 x 4
def select_pairs_by_height_dif(box_pairs, height_dif_thresh = 3):
    h1    = box_pairs[:, 0, -1]
    h2    = box_pairs[:, 1, -1]
    inds  = np.where(np.abs(h1 - h2) > height_dif_thresh)[0]
    return box_pairs[inds, :, :]

def load_tracks_from_gt(tracking_file):

    # load tracking results from txt file
    frame_ids = []
    track_ids = []
    with open(tracking_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
            frame_ids.append(int(splits[0]))
            track_ids.append(int(splits[1]))

    # frame ids and tracks ids
    frame_ids = list(set(frame_ids))
    track_ids = list(set(track_ids))

    track_numpy = np.zeros((len(track_ids), len(frame_ids), 4), dtype = np.float32)
    print('track numpy shape: {}'.format(track_numpy.shape))
    track_ids_to_indices = {}
    for i, trk_id in enumerate(track_ids):
        track_ids_to_indices[trk_id] = i

    frame_ids_to_indices = {}
    for i, frame_id in enumerate(frame_ids):
        frame_ids_to_indices[frame_id] = i

    for line in lines:
        splits = line.split(',')
        cls    = int(splits[6])
        if not cls == 1: continue
        frame_id, track_id = int(splits[0]), int(splits[1])
        box  = np.array([float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5])])
        track_numpy[track_ids_to_indices[track_id], frame_ids_to_indices[frame_id], :] = box
    
    return track_numpy, track_ids

def load_tracks_from_tracking(tracking_file):

    # load tracking results from txt file
    frame_ids = []
    track_ids = []
    with open(tracking_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
            frame_ids.append(int(splits[0]))
            track_ids.append(int(splits[1]))

    # frame ids and tracks ids
    frame_ids = list(set(frame_ids))
    track_ids = list(set(track_ids))

    track_numpy = np.zeros((len(track_ids), len(frame_ids), 4), dtype = np.float32)
    print('track numpy shape: {}'.format(track_numpy.shape))
    track_ids_to_indices = {}
    for i, trk_id in enumerate(track_ids):
        track_ids_to_indices[trk_id] = i

    frame_ids_to_indices = {}
    for i, frame_id in enumerate(frame_ids):
        frame_ids_to_indices[frame_id] = i

    for line in lines:
        splits = line.split(',')
        # cls    = int(splits[6])
        # if not cls == 1: continue
        frame_id, track_id = int(splits[0]), int(splits[1])
        box  = np.array([float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5])])
        track_numpy[track_ids_to_indices[track_id], frame_ids_to_indices[frame_id], :] = box
    
    return track_numpy, track_ids
