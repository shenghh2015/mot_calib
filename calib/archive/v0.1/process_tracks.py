import os
from calib.get_box_infront import gen_dir
from calib.utils import get_front_bboxes, get_front_bboxes_nms
import numpy as np
import sys
import cv2 as cv
import json
import random
from utils import load_seq_info

from const import MOT17_TRAIN_SEQ_NAMES, MOT17_SEQ_NAMES
from yolox.camera.img_vis import create_draw_board, draw_boxes

from const import EPS

sys.path.insert(0, os.path.abspath(".."))
import utils
import argparse

random.seed(0)
nb_color = 10000
colors = np.random.randint(0,255,(nb_color,3))

def load_json(json_file):
    return json.load(open(json_file))

def gen_kernel(ksize):
    base_kernel = np.array([1, 2, 1])
    conv_times = (ksize - 3) // 2
    kernel = base_kernel.copy()
    for i in range(conv_times):
        kernel = np.convolve(kernel, base_kernel, 'full')
    return kernel.flatten()

def smooth_1d(vec, ksize = 3, mode = 'same'):
    kernel = gen_kernel(ksize)
    kernel = np.array(kernel)
    kernel = kernel / np.sum(kernel)
    vec = np.convolve(vec, kernel, mode = mode)
    return vec

def smooth_boxes(boxes, ksize = 3):
    boxes = boxes.reshape((-1, 4))  # N x 4
    # top left, bottom, right
    tl_br = boxes.copy()
    tl_br[:, 2:] += tl_br[:,:2]
    for i in range(4):
        coord = tl_br[:, i]
        s_coord = smooth_1d(coord, ksize = ksize)
        tl_br[:, i] = s_coord
    tl_br[:,2:] -= tl_br[:,:2]
    return tl_br

def clip_boxes(boxes, clip_sizes = [3000, 20000]):
    box_sizes = boxes[:, :, 2] * boxes[:, :, 3]
    lower_size, upper_size = clip_sizes
    mask = np.expand_dims(np.logical_and(box_sizes < upper_size,\
            box_sizes > lower_size), axis = -1)
    return mask

def erosion_1d(ind_mask, ksize = 13):
    new_mask = np.zeros((len(ind_mask), 1))
    left, right, p = 0, 0, 0
    while p < len(ind_mask) and p < len(ind_mask):
        if ind_mask[p]:
            left = p
            while p < len(ind_mask) and ind_mask[p]: p += 1
            right = p
            # process mask
            if left + ksize - 1 < right - (ksize - 1):
                new_mask[left + ksize - 1: right - (ksize - 1)] = 1
            else:
                new_mask[left: right] = 0
        else:
            p += 1
    return new_mask

def mask_erosion(mask, ksize = 13):
    new_mask = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        new_mask[i, :, :] = erosion_1d(mask[i, :, 0], ksize)
    return new_mask

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=str, default = 'MOT17-09')      
    args = parser.parse_args()
    return args

def get_front_mask(all_bboxes, h_img = 1080):
    output_mask = np.zeros(all_bboxes.shape[:2])
    # print(output_mask.shape)
    for frame_id in range(all_bboxes.shape[1]):
        inds = get_front_bboxes(all_bboxes[:, frame_id, :], h_img)
        output_mask[inds, frame_id] = 1.
    
    return np.expand_dims(output_mask, axis = -1)


# all_bboxes: num_tracks x num_frames
# output:     num_tracks x num_frames
def front_tag_screen(all_bboxes, h_img = 1080, front_ratio_thresh = 0.8):
    output_mask = np.zeros(all_bboxes.shape[:2])
    for frame_id in range(all_bboxes.shape[1]):
        inds = get_front_bboxes_nms(all_bboxes[:, frame_id, :], h_img)
        output_mask[inds, frame_id] = 1.

    box_mask    = all_bboxes[:, :, 2] * all_bboxes[:, :, 3] > 0
    track_lens  = box_mask.sum(axis = -1)

    front_times = output_mask.sum(axis = -1)

    front_ratio = front_times / (track_lens + EPS)

    back_mask   = front_ratio < front_ratio_thresh
    print(len(back_mask), np.sum(1 - back_mask))
    box_mask[back_mask, :] = False

    return np.expand_dims(box_mask, axis = -1)

def front_tag_screen2(all_bboxes, h_img = 1080):
    output_mask = np.zeros(all_bboxes.shape[:2])
    for frame_id in range(all_bboxes.shape[1]):
        inds = get_front_bboxes_nms(all_bboxes[:, frame_id, :], h_img)
        output_mask[inds, frame_id] = 1.
    return output_mask

    # box_mask    = all_bboxes[:, :, 2] * all_bboxes[:, :, 3] > 0
    # track_lens  = box_mask.sum(axis = -1)

    # front_times = output_mask.sum(axis = -1)

    # front_ratio = front_times / (track_lens + EPS)

    # back_mask   = front_ratio < front_ratio_thresh
    # print(len(back_mask), np.sum(1 - back_mask))
    # box_mask[back_mask, :] = False

    # return np.expand_dims(box_mask, axis = -1)

def main():
    root_dir  = os.path.abspath('../')
    args = get_params()
    dataset   = 'mot'

    # seq_name = 'MOT17-04-FRCNN'

    for seq_name in MOT17_SEQ_NAMES:
        dataset_dir  = root_dir + f'/datasets/{dataset}'

        proj_name = 'calib'
        vis_dir   = os.path.join(root_dir, f'results/{proj_name}/{dataset}/{seq_name}')

        tracks    = np.load(vis_dir + '/tracks.npy')
        track_ids = np.load(vis_dir + '/track_ids.npy')

        # save all tracks in numpy array
        numpy_tracks  = tracks.copy()

        seq_info_file = os.path.join(dataset_dir, 'train', seq_name, 'seqinfo.ini') if \
                        seq_name in MOT17_TRAIN_SEQ_NAMES else os.path.join(dataset_dir, 'test', seq_name, 'seqinfo.ini')
    
        seq_info      = load_seq_info(seq_info_file)
        
        print(seq_info)
        w_img, h_img = seq_info['w_img'], seq_info['h_img']

        lower_size, upper_size = 1000, 200000

        # use only the front tracks
        # if seq_name == 'MOT17-05-FRCNN':
        #     front_ratio = 0.4
        # else:
        #     front_ratio = 0.8

        # front_mask   = front_tag_screen(numpy_tracks, h_img = h_img, front_ratio_thresh = front_ratio)
        # numpy_tracks = numpy_tracks * front_mask
        front_mask   = front_tag_screen2(numpy_tracks, h_img = h_img)
        np.save(vis_dir + '/front_mask.npy', front_mask)

        mask  = clip_boxes(numpy_tracks, clip_sizes = [lower_size, upper_size])
        # 1d eroson
        ksize = 13
        mask  = mask_erosion(mask, ksize = ksize)

        # print(mask.shape, front_mask.shape)

        board = create_draw_board(w_img, h_img)

        mask_dir = vis_dir + '/masks'
        gen_dir(mask_dir)
        
        visualize_track = True
        # smooth track
        s_tracks = numpy_tracks.copy()
        for i in range(numpy_tracks.shape[0]):
            draw_board = board.copy()
            boxes = numpy_tracks[i, :, :]
            s_tracks[i, :, :] = smooth_boxes(boxes, ksize = ksize)  # smooth boxes for each track
            if visualize_track:
                if front_mask[i, :].sum() > 0:
                    draw_boxes(draw_board, s_tracks[i, :, :], colors[track_ids[i]].tolist(), thickness = 2)
                    cv.imwrite(mask_dir + '/mask_{}.png'.format(track_ids[i]), draw_board)

        # mask track
        m_tracks = s_tracks * mask

        if visualize_track:
            draw_board = board.copy()
            for i in range(numpy_tracks.shape[0]):
                if front_mask[i, :].sum() > 0:
                    draw_board = board.copy()                            
                    draw_boxes(draw_board, m_tracks[i, :, :], colors[track_ids[i]].tolist(), thickness = 2)
                    cv.imwrite(mask_dir + '/sm_mask_{}.png'.format(track_ids[i]), draw_board)

        np.save(vis_dir + '/masked_tracks.npy', m_tracks)

if __name__ == '__main__':
    main()