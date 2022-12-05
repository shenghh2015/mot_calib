import os
from tools.box_utils import gen_front_mask, smoothen_tracks, smoothen_tracks_v2
import numpy as np 
import cv2 as cv
import sys
import os

# 
EPS = 1e-10

# init ECC module
number_of_iterations = 50
termination_eps = 0.01
# Define termination criteria
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

class BoxDataLoader():

    def __init__(self, track_file         = '.',
                       img_size           = (1920, 1080),
                       window             = 60,      # size of time window
                       stride             = 30,      # stride of time window
                       height_dif_thresh  = 2,       # threshold for height difference (image plane)
                       front_ratio_thresh = 0.8,     # threshold for the ratio of being as front tracks
                       fps                = 30,
                       use_clip           = False,
                       use_front          = True,
                       use_aspect_ratio   = False,
                       use_ecc            = False,
                       multi_ecc          = False,
                       img_dir            = '.',
                       time_window        = 3,
                       num_frames         = 400):

        self.window             = window                    # time window size
        self.stride             = stride                    # stride for sliding window
        self.front_ratio_thresh = front_ratio_thresh
        self.height_dif_thresh  = height_dif_thresh
        self.fps                = fps
        self.use_clip           = use_clip
        self.use_front          = use_front
        self.use_aspect_ratio   = use_aspect_ratio
        self.img_dir            = img_dir
        self.time_window        = time_window
        self.h_img              = img_size[1]

        self.track_file         = track_file
        self.num_frames         = num_frames

        # # load tracks from tracking file
        # self.tracks, self.track_ids, self.front_masks = self.load_tracks()

        # if self.num_frames < self.tracks.shape[1]:
        #     self.tracks      = self.tracks[:, :self.num_frames, :]
        #     self.track_ids   = self.track_ids[:self.num_frames]
        #     self.front_masks = self.front_masks[:, :self.num_frames]

        # # use ecc
        # if use_ecc:
        #     if multi_ecc:
        #         self.ecc_coefs = self.gen_ecc_coefficients_v2(fps = self.fps, time_window = self.time_window)
        #     else:
        #         self.ecc_coefs = self.gen_ecc_coefficients(fps = self.fps,    time_window = self.time_window)

        #     self.tracks    = self.map_to_ref(self.ecc_coefs, self.tracks, dsample = 4)

        # # smoothen tracks
        # self.tracks = smoothen_tracks_v2(self.tracks, use_clip = self.use_clip)

        
        # self.box_pairs          = self.gen_box_pairs()

        self.raw_tracks, _      = self.load_raw_tracks()


    def load_raw_tracks(self):
        # load tracks from txt file
        base_name = os.path.basename(self.track_file)

        if base_name.endswith('gt.txt'):
            track_boxes, track_ids = load_tracks_from_gt(self.track_file)
        else:
            track_boxes, track_ids = load_tracks_from_tracking(self.track_file)

        return track_boxes, track_ids

    def ecc(self, frame, ref, dsample = 4):
        h, w              = frame.shape[:2]
        h, w              = h // dsample, w // dsample
        frame             = cv.resize(frame, (w, h))
        ref               = cv.resize(ref,   (w, h))
        warp_matrix       = np.eye(3, 3, dtype=np.float32)
        gray_frame        = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ref_frame         = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
        try:
            (cc, warp_matrix) = cv.findTransformECC(gray_frame, ref_frame, warp_matrix, cv.MOTION_HOMOGRAPHY, criteria)
            return warp_matrix
        except: 
            return warp_matrix

    def get_ecc_warpped_boxes(self, start_frame_id, frame_window, use_ecc = True):

        # compute ecc coefficients
        ref_frame_id = start_frame_id + frame_window // 2
        ref_frame    = cv.imread(self.img_dir + '/{:06d}.jpg'.format(ref_frame_id + 1))
        ecc_coefs    = np.tile(np.eye(3, 3, dtype = np.float32).reshape((1, 3, 3)), (frame_window, 1, 1))

        if use_ecc:
            for frame_id in range(start_frame_id, start_frame_id + frame_window):
                if frame_id == ref_frame_id: continue
                frame    = cv.imread(self.img_dir + '/{:06d}.jpg'.format(frame_id + 1))
                ecc_coef = self.ecc(frame, ref_frame)
                ecc_coefs[frame_id - start_frame_id, :, :] = ecc_coef

        # warp boxes to reference frame
        tracks  = self.raw_tracks[:, start_frame_id: start_frame_id + frame_window, :]
        tracks  = self.map_to_ref(ecc_coefs, tracks)

        # smoothen boxes
        tracks  = smoothen_tracks_v2(tracks, ksize = 13, use_clip = self.use_clip)
        
        # print(tracks[:2, :])
        front_masks = self.front_masks[:, start_frame_id: start_frame_id + frame_window]

        # compute height pairs
        num_batchs = (frame_window - self.window) // self.stride

        # print(tracks.shape, num_batchs)
        batchs = []
        for i in range(num_batchs):

            clip_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

            for clip_ratio in clip_ratios:
                
                # end_frame   = start_frame_id + int(i * self.stride + self.window * clip_ratio)
                s_frame_id  = i * self.stride
                e_frame_id  = int(i * self.stride + self.window * clip_ratio)

                # print(s_frame_id, e_frame_id)
                track_boxes = tracks[:, s_frame_id: e_frame_id, :]
                
                # track_boxes = self.tracks[:, start_frame_id + i * self.stride:\
                #     end_frame, :]  # N x W x 4
                
                if self.use_front:
                    # a mask of front tracks
                    # front_mask = self.front_masks[:, start_frame_id + i * self.stride:\
                    #     end_frame]  # N x W x 4
                    front_mask = front_masks[:, s_frame_id: e_frame_id]
                else:
                    # a mask of front tracks
                    front_mask = np.ones(track_boxes.shape[:2])
                
                # select continous and high-front-ratio tracks
                ind_mask = get_ind_mask(track_boxes[:,:,-1], front_mask) 
                masked_track_boxes = track_boxes[ind_mask, :, :]
                box_pairs = masked_track_boxes[:, [0, - 1], :]
                
                # print(box_pairs.shape)
                
                # reject pairs of boxes that height difference is small
                box_pairs = select_pairs_by_height_dif(box_pairs, height_dif_thresh = self.height_dif_thresh)

                if self.use_aspect_ratio:
                    box_pairs = select_pairs_by_aspect_ratio(box_pairs, asp_dif_thresh = 0.05)

                batchs.append(box_pairs)

        return np.concatenate(batchs, axis = 0)

    def gen_ecc_coefficients(self, fps = 30, time_window = 3):

        frame_window = int(fps * time_window)
        num_frames   = self.tracks.shape[1]
        ecc_coefs    = np.tile(np.eye(3, 3).reshape((1, 3, 3)), (num_frames, 1, 1))
        for frame_id in range(num_frames):
            print(f'frame id : {frame_id}')
            ind_window = frame_id // frame_window
            if frame_id == ind_window * frame_window + frame_window // 2: continue
            if frame_id % frame_window == 0:
                ref_frame    = cv.imread(self.img_dir + '/{:06d}.jpg'.format(frame_id + frame_window // 2))
            frame = cv.imread(self.img_dir + '/{:06d}.jpg'.format(frame_id + 1))
            # ecc
            ecc_coef  = self.ecc(frame, ref_frame)
            ecc_coefs[frame_id, :] = ecc_coef
        
        return ecc_coefs

    def gen_ecc_coefficients_v2(self, fps = 30, time_window = 3):

        frame_window = int(fps * time_window)
        num_frames   = self.tracks.shape[1]
        ecc_coefs    = np.tile(np.eye(3, 3).reshape((1, 3, 3)), (num_frames, 1, 1))
        for frame_id in range(num_frames):
            print(f'frame id : {frame_id}')
            ind_window   = frame_id // frame_window
            ref_frame_id = ind_window * frame_window + frame_window // 2
            if frame_id == ref_frame_id: continue
            if frame_id % frame_window == 0:
                ref_frame     = cv.imread(self.img_dir + '/{:06d}.jpg'.format(ref_frame_id + 1))
            frame = cv.imread(self.img_dir + '/{:06d}.jpg'.format(frame_id + 1))

            # middle reference
            mid_ref_id    = (frame_id + ref_frame_id) // 2
            mid_ref_frame = cv.imread(self.img_dir + '/{:06d}.jpg'.format(mid_ref_id + 1))
            
            # first ecc
            ecc_coef1 = self.ecc(frame, mid_ref_frame)

            # second ecc
            ecc_coef2 = self.ecc(mid_ref_frame, ref_frame)

            ecc_coefs[frame_id, :] = np.matmul(ecc_coef2, ecc_coef1)

        return ecc_coefs

    def map_to_ref(self, ecc_coefs, tracks, dsample = 4):
        EPS = 1e-10

        if len(ecc_coefs) == 2: ecc_coefs = ecc_coefs.reshape((-1, 3, 3))

        tracks    = tracks.copy() / dsample
        ecc_coefs = np.expand_dims(ecc_coefs, axis = 0)
        ecc_coefs = np.tile(ecc_coefs, (tracks.shape[0], 1, 1, 1))
        
        # track shape
        n, f   = tracks.shape[:2]

        mask   = tracks[:, :, -1] > 0

        tracks = tracks.reshape((-1, 4))

        t         = tracks[:, :2].copy()
        t[:, 0]  += tracks[:, 2] / 2
        
        r         = tracks[:, :2].copy()
        r[:, 0]  += tracks[:, 2]
        r[:, 1]  += tracks[:, 3] / 2

        b         = tracks[:, :2].copy()
        b[:, 0]  += tracks[:, 2] / 2
        b[:, 1]  += tracks[:, 3]

        l         = tracks[:, :2].copy()
        l[:, 1]  += tracks[:, 3] / 2

        trbl      = np.stack([t, r, b, l], axis = 0)
        trbl      = trbl.reshape((-1, 2))
        trbl      = np.concatenate([trbl, np.ones((4 * n * f, 1))], axis = -1)  # n * f * 4, 3
        trbl      = np.expand_dims(trbl, axis = -1)
        
        new_coefs = np.tile(ecc_coefs, (4, 1, 1)).reshape((-1, 3, 3))         # n * f * 4, 3, 3

        new_trbl  = np.matmul(new_coefs, trbl)                                # n * f * 4, 3, 1
        
        new_trbl  = new_trbl.squeeze()
        new_trbl  = new_trbl[:, :2] / (new_trbl[:, 2:] + EPS)

        new_trbl  = new_trbl.reshape((4, n, f, 2))
        new_trbl  = new_trbl.transpose((1, 2, 0, 3))

        xmin, xmax  = new_trbl[:, :, :, 0].min(axis = -1), new_trbl[:, :, :, 0].max(axis = -1)
        ymin, ymax  = new_trbl[:, :, :, 1].min(axis = -1), new_trbl[:, :, :, 1].max(axis = -1)

        new_tracks  = np.zeros(new_trbl.shape[:3])
        ecc_tracks  = new_tracks.copy()
        new_tracks[:, :, 0] = xmin
        new_tracks[:, :, 1] = ymin
        new_tracks[:, :, 2] = xmax - xmin
        new_tracks[:, :, 3] = ymax - ymin

        ecc_tracks[mask]    = new_tracks[mask]

        ecc_tracks = ecc_tracks * dsample

        return ecc_tracks

    def load_tracks(self):

        # load tracks from txt file
        base_name = os.path.basename(self.track_file)

        if base_name.endswith('gt.txt'):
            track_boxes, track_ids = load_tracks_from_gt(self.track_file)
        else:
            track_boxes, track_ids = load_tracks_from_tracking(self.track_file)

        # generate front mask to indicate if a track is front or not at each frame
        front_masks     = gen_front_mask(track_boxes, h_img = 1080)

        return track_boxes,  np.array(track_ids), front_masks
 
    def get_ids(self):
        return self.track_ids
    
    def get_all_pairs(self):
        return self.box_pairs

    def gen_box_pairs(self):
        seq_len = self.tracks.shape[1]
        num_batchs = (seq_len - self.window) // self.stride
        batchs = []
        for i in range(num_batchs):

            clip_ratios = [0.5, 0.6, 0.8, 1]

            for clip_ratio in clip_ratios:
                
                end_frame = int(i * self.stride + self.window * clip_ratio)
                # track boxes in a time window
                track_boxes = self.tracks[:, i * self.stride:\
                    end_frame, :]  # N x W x 4
                
                if self.use_front:
                    # a mask of front tracks
                    front_mask = self.front_masks[:, i * self.stride:\
                        end_frame]  # N x W x 4
                else:
                    # a mask of front tracks
                    front_mask = np.ones(track_boxes.shape[:2])
                
                # select continous and high-front-ratio tracks
                ind_mask = get_ind_mask(track_boxes[:,:,-1], front_mask) 
                masked_track_boxes = track_boxes[ind_mask, :, :]
                box_pairs = masked_track_boxes[:, [0, - 1], :]
                
                # reject pairs of boxes that height difference is small
                box_pairs = select_pairs_by_height_dif(box_pairs, height_dif_thresh = self.height_dif_thresh)

                if self.use_aspect_ratio:
                    box_pairs = select_pairs_by_aspect_ratio(box_pairs, asp_dif_thresh = 0.05)

                batchs.append(box_pairs)
        
        self.batchs = batchs

        return np.concatenate(self.batchs, axis = 0)

    def get_boxe_pairs_by_window(self, start_frame, window):

        num_batchs = (window - self.window) // self.stride
        batchs = []
        for i in range(num_batchs):

            clip_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

            for clip_ratio in clip_ratios:
                
                end_frame = start_frame + int(i * self.stride + self.window * clip_ratio)
                track_boxes = self.tracks[:, start_frame + i * self.stride:\
                    end_frame, :]  # N x W x 4
                
                if self.use_front:
                    # a mask of front tracks
                    front_mask = self.front_masks[:, start_frame + i * self.stride:\
                        end_frame]  # N x W x 4
                else:
                    # a mask of front tracks
                    front_mask = np.ones(track_boxes.shape[:2])
                
                # select continous and high-front-ratio tracks
                ind_mask = get_ind_mask(track_boxes[:,:,-1], front_mask) 
                masked_track_boxes = track_boxes[ind_mask, :, :]
                box_pairs = masked_track_boxes[:, [0, - 1], :]
                
                # reject pairs of boxes that height difference is small
                box_pairs = select_pairs_by_height_dif(box_pairs, height_dif_thresh = self.height_dif_thresh)

                if self.use_aspect_ratio:
                    box_pairs = select_pairs_by_aspect_ratio(box_pairs, asp_dif_thresh = 0.05)

                batchs.append(box_pairs)

        return np.concatenate(batchs, axis = 0)

    def get_batch_counts(self):
        return len(self.batchs)
    
    def get_batch(self, batch_index):
        return self.batchs[batch_index]

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

def select_pairs_by_aspect_ratio(box_pairs, asp_dif_thresh = 0.05):
    a1   = box_pairs[:, 0, 3] / box_pairs[:, 0, 2]
    a2   = box_pairs[:, 1, 3] / box_pairs[:, 1, 2]
    inds = np.where(np.abs((a1 - a2) / a2) < asp_dif_thresh)[0]
    # print(inds)
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
