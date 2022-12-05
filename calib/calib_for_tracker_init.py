'''
    Use a short time of frames (e.g. in first 3 secs) to calbirate the camera
'''
from geometry import gen_focal_length
from camera_utils import Camera, AerialView
from geometry_3d import Rotation
from calib_losses import constrained_alpha_beta_fov_loss
import numpy as np
import os
import cv2
from scipy import optimize
import argparse
import json

from visualize import create_black_board, draw_text, vis_xyz_as_images, vis_box_pairs, plot_height_stats
from datasets_init import BoxDataLoader
from utils import load_seq_info, save_to_json


from const import MOT17_SEQ_NAMES, MOT17_TRAIN_SEQ_NAMES, MOT20_TRAIN_SEQ_NAMES, RANDOM_COLORS, MOT17_STATIC_SEQ_NAMES
from yolox.camera.img_vis import draw_box, draw_corners, draw_boxes, draw_text

import time

def gen_dir(folder):
    if not os.path.exists(folder): 
        os.system(f'mkdir -p {folder}')

def get_box_pts(boxes, 
                loc = 'bottom'):
    if loc == 'center':
        box_pts = boxes[:, :2] + boxes[:, 2:] / 2
    elif loc == 'bottom':
        box_pts = boxes[:, :2].copy()
        box_pts[:, 0] += boxes[:, 2] / 2 # x + w/2
        box_pts[:, 1] += boxes[:, 3]     # y + h
    return box_pts

def get_xy_ranges(a_view, tracks):

    boxes = tracks.reshape((-1, 4))
    mask  = boxes[:, 2] * boxes[:, 3] > 0
    boxes = boxes[mask]

    xyh = a_view.compute_xyh_from_boxes(boxes)
    xmin, xmax = xyh[:, 0].min(), xyh[:, 0].max()
    ymin, ymax = xyh[:, 1].min(), xyh[:, 1].max()

    return [xmin, ymin, xmax, ymax]


def dec_calib(boxes, bounds, a_view, 
              use_core = True):

    alpha, beta, fov = bounds

    new_boxes = boxes.copy()
    if use_core:
        centers = boxes.copy()
        centers[:, :, :2] += boxes[:, :, 2:] / 2
        centers[:, :, 2]   = centers[:, :, 3] / 4
        centers[:, :, :2] -= centers[:, :, 2:] / 2
        new_boxes          = centers

    params  = (a_view, new_boxes)
    result  = optimize.differential_evolution(constrained_alpha_beta_fov_loss, [alpha, beta, fov], args = params, 
                    updating='deferred', tol = 1e-10, workers=20, disp = False, popsize=50)

    alpha, beta, fov = result.x

    return np.array([alpha, beta, fov]), result.fun

def compute_heights(a_view, tracks):

    boxes = tracks.reshape((-1, 4))
    mask  = boxes[:, 2] * boxes[:, 3] > 0
    boxes = boxes[mask]

    heights = a_view.compute_heights_from_boxes(boxes)

    return heights.squeeze()

def robust_stats(heights, num_iters = 1, num_std = 2):
    mask = np.ones((len(heights), )) > 0

    for it in range(num_iters):
        
        # mean and std
        h = heights[mask]
        mu, std = h.mean(), h.std()

        # update mask
        mask = np.logical_and(heights > mu - num_std * std, heights < mu + num_std * std)

    return heights[mask]

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default = 'mot')
    parser.add_argument("--subset",       type=str, default = 'train')    
    parser.add_argument("--seq_name",     type=str, default = 'MOT17-13-FRCNN')
    parser.add_argument("--use_gt",       action="store_true", help="use ground truth or not")
    parser.add_argument("--workers",      type=int, default = 20)
    parser.add_argument("--popsize",      type=int, default = 50) 
    parser.add_argument("--start",        type=int, default = 0)
    parser.add_argument("--time_window",       type=int, default = 3)
    args = parser.parse_args()
    return args

def main():

    args     = get_params()
    root_dir = os.path.abspath('../')       
    # dataset_name = 'mot'     
    # seq_name     = 'MOT17-10-FRCNN'     
    dataset_name   = args.dataset_name          
    seq_name       = args.seq_name
    start_frame_id = args.start

    use_gt       = False
    use_core     = True

    if seq_name in MOT17_STATIC_SEQ_NAMES:
        use_ecc  = False
    else:
        use_ecc  = True
    
    if seq_name in MOT17_TRAIN_SEQ_NAMES:
        subset   = 'train'
    else:
        subset   = 'test'

    time_window  = args.time_window
    num_frames   = 750

    print('------------------------------------------------\n')
    seq_info_file = os.path.join(root_dir, 'datasets', 
                        dataset_name, subset,  seq_name, 'seqinfo.ini')
    seq_info = load_seq_info(seq_info_file)
    print(seq_info)

    calib_res_dir = os.path.join(root_dir, 'calib_init_results', dataset_name)
    gen_dir(calib_res_dir)

    vis_res_dir  = os.path.join(root_dir, 'calib_init_results', dataset_name, 'vis_results')
    gen_dir(vis_res_dir)

    # intialize aerial view objects for calibration
    w_img, h_img = seq_info['w_img'], seq_info['h_img']
    rotation     = Rotation(np.array([0, 0, 0]))
    camera       = Camera(w_img = w_img, h_img = h_img, 
                          cx = w_img / 2, cy = h_img / 2)
    a_view       = AerialView(camera, rotation)

    # load data for calibration
    if use_gt:
        track_file = os.path.join(root_dir, 'datasets', dataset_name, subset, seq_name, 'gt/gt.txt')
    else:
        if seq_name in MOT17_SEQ_NAMES:
            track_file = os.path.join(root_dir, 'results', f'mot17_paper_v3/bytetrack/{seq_name}.txt')
        else:
            track_file = os.path.join(root_dir, 'results', f'mot20_paper/yolox_x_mix_mot20_ch/bytetrack/{seq_name}.txt')

    boxloader    = BoxDataLoader(track_file = track_file,
                                img_size           = (w_img, h_img),
                                window             = int(seq_info['fps'] * 2),
                                stride             = seq_info['fps'] // 2,
                                height_dif_thresh  = 3,
                                front_ratio_thresh = 0.8,
                                fps                = seq_info['fps'],
                                use_ecc            = False,
                                multi_ecc          = False,
                                img_dir            = os.path.join(root_dir, 'datasets', dataset_name, subset, seq_name, 'img1'),
                                time_window        = time_window,
                                num_frames         = num_frames)

    # start_frame_id = 0
    frame_window   = time_window * seq_info['fps']
    end_frame_id   = start_frame_id + frame_window
    ref_frame_id   = start_frame_id
    boxes          = boxloader.get_ecc_warpped_boxes_v2(start_frame_id, frame_window, 
                                                        ref_frame_id = 0, use_ecc = use_ecc)
    print(f'samples: {tuple(boxes.shape)}')

    bounds         = ((20, 100), (-10, 10), (30, 150))
    result, loss_val = dec_calib(boxes, bounds, a_view, use_core = use_core)
    alpha, beta, fov = result
    alpha, beta, fov = float(alpha), float(beta), float(fov)
    print(f'start_frame_id: {start_frame_id}, end_frame_id: {end_frame_id}, ref_frame_id:{ref_frame_id} time: {time_window} secs')
    print(f'alpha: {round(alpha, 2)}, beta: {round(beta, 2)}, fov: {round(fov, 2)}, loss:{round(loss_val, 6)}')

    # save calibration results
    calib_info = {'w_img': w_img, 'h_img': h_img, 'alpha': alpha, 'beta':  beta, 'fov': fov, 'loss': loss_val, 'samples': tuple(boxes.shape)}

    # compute height statistics
    all_tracks      = boxloader.raw_tracks.copy()
    track_mask      = all_tracks[:, :, -1] > 0
    all_tracks      = all_tracks[track_mask, :]
    heights         = compute_heights(a_view, all_tracks)
    print(f'number of heights: {len(heights)}')
    heights         = robust_stats(heights, num_iters = 5, num_std = 3)
    print(f'number of heights: {len(heights)}')
    vis_height_stat = plot_height_stats(heights, 
                                        labels = ['heights', 'counts'],
                                        title = f'{seq_name}')
    cv2.imwrite(vis_res_dir + f'/{seq_name}_height_stats.png', vis_height_stat)
    calib_info['h_mu']  = heights.mean()
    calib_info['h_std'] = heights.std()
    
    save_to_json(calib_res_dir + f'/{seq_name}.json', calib_info)

if __name__ == '__main__':
    main()