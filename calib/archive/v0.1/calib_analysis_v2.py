from tkinter import S
from camera_utils import Camera, AerialView
from calib.geometry import compute_xyh_from_boxes, fov_to_s, s_to_fov
from calib.geometry_3d import Rotation
from calib_losses import alpha_beta_s_loss, alpha_beta_loss, s_loss
import numpy as np
import os
import cv2
from scipy import optimize

from visualize import create_black_board, plot_loss_1d, vis_box_pairs, vis_xyz_as_images
from datasets import BoxDataLoader
from utils import load_seq_info, analyze_s_loss, analyze_alpha_loss, analyze_beta_loss, analyze_s_loss_v2, analyze_loss

from const import INITIAL_FOVS, MOT17_TRAIN_SEQ_NAMES, MOT17_STATIC_SEQ_NAMES, RANDOM_COLORS
from yolox.camera.img_vis import draw_box, draw_corners
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

def get_val_rngs(cam_info, tracks, lower_ratio = 0.01, upper_ratio = 99.9):
    # print(tracks.shape)
    boxes    = tracks.reshape((-1, 4))
    # print(boxes.shape)
    box_mask = boxes[:, 2] * boxes[:, 3] > 0
    # print(box_mask.sum())
    boxes = boxes[box_mask, :]
    xyh   = compute_xyh_from_boxes(cam_info, boxes)

    x, y, h = xyh[:, 0], xyh[:, 1], xyh[:, 2]
    # print(x.mean(), x.std(), y.mean(), y.std())
    # print(x, y, h)
    val_rngs = []
    for z in [x, y, h]:
        val_rng = max(abs(np.percentile(z, lower_ratio)),
                           abs(np.percentile(z, upper_ratio)))
        val_rngs.append(val_rng)
    return val_rngs

def get_inital_fov(seq_name):

    if seq_name in INITIAL_FOVS:
        return INITIAL_FOVS[seq_name]
    else:
        return 65

# boxes: N x 2 x 4
# initals: [init_alpha, init_beta, init_fov]
# return: alpha, beta, fov
def iteractive_calib(boxes, initials, cam_params, num_inters, disp = True):

    alpha, beta, fov = initials
    s = fov_to_s(fov)
    
    for it in range(num_inters):

        print(f'Iterations: {it}')
        
        params  = (cam_params, boxes, s)
        result  = optimize.minimize(alpha_beta_loss, [alpha, beta], args = params, 
                        method = 'BFGS', options={'gtol': 1e-10, 'disp': disp})

        alpha, beta = result.x

        if disp: print(result)

        # s       = fov_to_s(80)
        # print('*************************')
        # print(f'inital s: {s}')

        params  = (cam_params, boxes, alpha, beta)
        result  = optimize.minimize(s_loss, [s], args = params, 
                        method = 'BFGS', options={'gtol': 1e-10, 'disp': disp}) 

        s       = result.x

        if disp: print(result)

    fov = s_to_fov(s)
    return np.array([alpha, beta, fov]), result.fun

def joint_calib(boxes, initials, cam_params, disp = True):

    alpha, beta, fov = initials

    s = fov_to_s(fov)

    params  = (cam_params, boxes)
    result  = optimize.minimize(alpha_beta_s_loss, [alpha, beta, s], args = params, 
                    method = 'BFGS', options={'gtol': 1e-10, 'disp': disp})

    alpha, beta, s = result.x

    if disp: print(result)

    fov = s_to_fov(s)

    return np.array([alpha, beta, fov]), result.fun

def get_disp_values(vals):
    numpy_vals = np.array(vals)
    return [round(float(numpy_vals[i]), 2) for i in range(len(numpy_vals))]

def main():

    root_dir = os.path.abspath('../')
    proj_name    = 'calib'
    dataset_name = 'mot'
    seq_name     = 'MOT17-09-FRCNN'

    # calib_method = 'joint'
    calib_method = 'iter'

    seq_set = MOT17_STATIC_SEQ_NAMES

    print('------------------------------------------------\n')
    if seq_name in MOT17_TRAIN_SEQ_NAMES:
        seq_info_file = os.path.join(root_dir, 'datasets',
                        dataset_name, 'train', seq_name, 'seqinfo.ini')
    else:
        seq_info_file = os.path.join(root_dir, 'datasets', 
                        dataset_name, 'test',  seq_name, 'seqinfo.ini')
    seq_info = load_seq_info(seq_info_file)
    print(seq_info)

    w_img, h_img = seq_info['w_img'], seq_info['h_img']

    use_gt = True
    use_one_box = False
    analyze_pairs = True

    if use_gt:
        track_file = os.path.join(root_dir, 'datasets', dataset_name, 'train', seq_name, 'gt/gt.txt')
    else:
        track_file = os.path.join(root_dir, 'results', proj_name, dataset_name)

    boxloader    = BoxDataLoader(track_file = track_file,
                                img_size    = (w_img, h_img),
                                window      = int(seq_info['fps'] * 2),
                                stride      = seq_info['fps'] // 2,
                                height_dif_thresh  = 3,
                                front_ratio_thresh = 0.85,
                                fps         = seq_info['fps'])
    boxes       = boxloader.get_all_pairs() # N x 2 x 4

    if use_one_box: boxes = boxes[:1, :, :]

    if analyze_pairs:
        track_dir = os.path.join(root_dir, 'results', proj_name, dataset_name, seq_name)
        vis_board = vis_box_pairs(w_img, h_img, boxes)
        cv2.imwrite(track_dir + '/box_distributioin.png', vis_board)

    print(f'samples: {boxes.shape}')

    cam_params = {'w_img': w_img, 'h_img': h_img}

    params = (cam_params, boxes)

    # num_pts = (81, 21, 91)
    num_pts = (161, 41, 181)
    bounds  = [(20, 100), (-10, 10), (fov_to_s(120), fov_to_s(30))]
    
    start_time = time.time()
    res, loss = analyze_loss(bounds, num_pts, params)
    alpha, beta, s = res
    fov  = s_to_fov(s)
    end_time   = time.time()
    duration = end_time - start_time
    print(f'global minimum: alpha {round(alpha, 2)}, beta {round(beta, 2)}, fov {round(fov, 2)} (s: {round(s, 2)}), loss:{round(loss,4)} dur: {round(duration, 2)}')



if __name__ == '__main__':
    main()