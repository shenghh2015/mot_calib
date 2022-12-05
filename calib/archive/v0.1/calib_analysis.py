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
from utils import load_seq_info, analyze_s_loss, analyze_alpha_loss, analyze_beta_loss, analyze_s_loss_v2

from const import INITIAL_FOVS, MOT17_TRAIN_SEQ_NAMES, MOT17_STATIC_SEQ_NAMES, RANDOM_COLORS
from yolox.camera.img_vis import draw_box, draw_corners

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

    init_fov    = get_inital_fov(seq_name)

    init_fov = 65
    # initials = [70, 0, init_fov]
    initial_angles = [70, 0]

    init_s       = s_to_fov(init_fov)
    params  = (cam_params, boxes, init_s)
    result  = optimize.minimize(alpha_beta_loss, initial_angles, args = params, 
                    method = 'BFGS', options={'gtol': 1e-10, 'disp': True})
    alpha, beta = result.x
    alpha, beta = float(alpha), float(beta)
    
    print(f'inital alpha, inital beta, fixed fov: {get_disp_values(initial_angles + [init_fov])}(s:{round(init_s, 2)})')
    print(f'optimized alpha, beta: {get_disp_values([alpha, beta])}')

    # analyze fov losses
    calib_res_dir = os.path.join(root_dir, 'results', proj_name, dataset_name, seq_name, 'calib')
    gen_dir(calib_res_dir)

    params = (cam_params, boxes)
    alpha_range = [0, 100]
    alpha_losses, alpha_space = analyze_alpha_loss(alpha_range, beta, init_s, 
                            alpha_range[1] - alpha_range[0] + 1, params)
    
    plot_loss_1d(calib_res_dir + f'/a-{alpha_range}_b{round(beta, 2)}_fov{round(init_fov, 2)}(s:{round(init_s, 2)})_loss_one_{use_one_box}.png', alpha_losses, alpha_space, 
                title = 'alpha loss', labels = ['alpha', 'height loss'],
                use_clip = True)
    min_ind = np.argmin(alpha_losses)
    print(f'minimum loss: {alpha_losses[min_ind]}, alpha: {alpha_space[min_ind]}')

    beta_range = [-10, 10]
    beta_losses, beta_space = analyze_beta_loss(alpha, beta_range, init_s, 
                            beta_range[1] - beta_range[0] + 1, params)
    plot_loss_1d(calib_res_dir + f'/a{round(alpha, 2)}_b-{beta_range}_fov{round(init_fov, 2)}(s:{round(init_s, 2)})_loss_one_{use_one_box}.png', beta_losses, beta_space, 
                title = 'beta loss', labels = ['beta', 'height loss'],
                use_clip = True)
    min_ind = np.argmin(beta_losses)
    print(f'minimum loss: {beta_losses[min_ind]}, beta: {beta_losses[min_ind]}')

    init_fov = 80
    init_s   = fov_to_s(init_fov)
    # print(f'inital s:{round(init_s, 2)}')
    params   = (cam_params, boxes, alpha, beta)
    result   = optimize.minimize(s_loss, [init_s], args = params, 
                    method = 'CG', options={'gtol': 1e-30, 'disp': True}) 

    print(result)
    s       = result.x
    fov = s_to_fov(s)
    fov = float(fov)
    s   = float(s)

    print(f'fixed alpha, fixed beta, inital fov: {get_disp_values([alpha, beta, init_fov])}(s:{round(init_s, 2)})')
    print(f'optimized fov: {get_disp_values([fov])}(s:{round(s, 2)})')

    s_range = [0.28, 1.8]
    params = (cam_params, boxes)
    s_losses, s_space = analyze_s_loss_v2(alpha, beta, s_range, 100, params)
    plot_loss_1d(calib_res_dir + f'/a{round(alpha, 2)}_b{round(beta, 2)}_s-{s_range}_loss_one_{use_one_box}.png', s_losses, s_space, 
                title = 's loss', labels = ['s', 'height loss'],
                use_clip = True)
    clip_x = [0.6,1.0]
    plot_loss_1d(calib_res_dir + f'/a{round(alpha, 2)}_b{round(beta, 2)}_s-{clip_x}_loss_one_{use_one_box}.png', s_losses, s_space, 
                title = 's loss', labels = ['s', 'height loss'],
                use_clip = True, clip_x = [0.6, 1.0])
    min_ind = np.argmin(s_losses)
    print(f'minimum loss: {s_losses[min_ind]}, s: {s_space[min_ind]}, fov: {s_to_fov(s_space[min_ind])}')
    print(f'fov range: {s_to_fov(clip_x[0]), s_to_fov(clip_x[1])}')

    # analyze s loss
    s_range = [0.27, 1.8]
    # alpha, beta = 90.13, -1.82
    alpha, beta = 91, -2
    params = (cam_params, boxes)
    s_losses, s_space = analyze_s_loss_v2(alpha, beta, s_range, 100, params)
    plot_loss_1d(calib_res_dir + f'/a{round(alpha, 2)}_b{round(beta, 2)}_s-{s_range}_loss_V2.png', s_losses, s_space, 
                title = 's loss', labels = ['s', 'height loss'],
                use_clip = True)
    # clip_x = [0.6,1.0]
    # plot_loss_1d(calib_res_dir + f'/a{round(alpha, 2)}_b{round(beta, 2)}_s-{clip_x}_loss_one_{use_one_box}.png', s_losses, s_space, 
    #             title = 's loss', labels = ['s', 'height loss'],
    #             use_clip = True, clip_x = [0.6, 1.0])
    min_ind = np.argmin(s_losses)
    print(f'minimum loss: {s_losses[min_ind]}, s: {s_space[min_ind]}, fov: {s_to_fov(s_space[min_ind])}')
    # print(f'fov range: {s_to_fov(clip_x[0]), s_to_fov(clip_x[1])}')

    # joint optimization
    # init_fov  = 65
    # init_s    = fov_to_s(init_fov)
    # params    = (cam_params, boxes)
    # initials  = [70, 0, init_s]
    # result    = optimize.minimize(alpha_beta_s_loss, initials, args = params, 
    #                 method = 'BFGS', options={'gtol': 1e-10, 'disp': True})
    # alpha, beta, s = result.x
    # alpha, beta, s = float(alpha), float(beta), float(s)
    # fov = s_to_fov(s)
    # print(f'optimized alpha, beta, s (fov): {get_disp_values([alpha, beta, s, fov])}')

if __name__ == '__main__':
    main()