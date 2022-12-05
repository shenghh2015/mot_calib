from tkinter import S
from camera_utils import Camera, AerialView
from calib.geometry import compute_xyh_from_boxes, fov_to_s, gen_focal_length, gen_focal_length_from_s, s_to_fov
from calib.geometry_3d import Rotation
from calib_losses import alpha_beta_s_loss, alpha_beta_loss, s_loss, alpha_beta_fov_loss
import numpy as np
import os
import cv2
from scipy import optimize

from visualize import create_black_board, draw_text, plot_loss_1d, vis_xyz_as_images
# from datasets import BoxDataLoader
from datasets import BoxDataLoader
from utils import load_seq_info, analyze_s_loss, analyze_alpha_loss, analyze_beta_loss

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

def de_calib(boxes, bounds, cam_params, disp = True):

    alpha, beta, fov = bounds

    # s = fov_to_s(fov)
    s = (fov_to_s(fov[1]), fov_to_s(fov[0]))

    params  = (cam_params, boxes)
    result  = optimize.differential_evolution(alpha_beta_s_loss, [alpha, beta, s], args = params, 
                    updating='deferred', tol = 1e-10, workers=4, disp = False, popsize=50)

    alpha, beta, s = result.x

    if disp: print(result)

    fov = s_to_fov(s)

    return np.array([alpha, beta, fov]), result.fun

def de_calib_v2(boxes, bounds, cam_params, disp = True):

    alpha, beta, fov = bounds

    # s = fov_to_s(fov)
    # s = (fov_to_s(fov[1]), fov_to_s(fov[0]))

    params  = (cam_params, boxes)
    result  = optimize.differential_evolution(alpha_beta_fov_loss, [alpha, beta, fov], args = params, 
                    updating='deferred', tol = 1e-10, workers=4, disp = False, popsize=50)

    alpha, beta, fov = result.x

    if disp: print(result)

    return np.array([alpha, beta, fov]), result.fun


def exhausive_calib(boxes, initials, cam_params, disp = True):

    alpha, beta, fov = initials
    # s = fov_to_s(fov)
    
    losses  = []
    results = []
    fov_space = np.linspace(30, 120, 91)

    for fov in fov_space:

        s       = fov_to_s(fov)
        params  = (cam_params, boxes, s)
        result  = optimize.minimize(alpha_beta_loss, [alpha, beta], args = params, 
                        method = 'BFGS', options={'gtol': 1e-10, 'disp': disp})

        alpha, beta = result.x

        if disp: print(result)

        losses.append(result.fun)
        results.append(result)
    
    min_ind = np.argmin(losses)

    min_loss     = losses[min_ind]
    alpha, beta  = results[min_ind].x
    fov          = fov_space[min_ind]

    # fov = s_to_fov(s)
    return np.array([alpha, beta, fov]), min_loss

def main():

    root_dir = os.path.abspath('../')
    proj_name    = 'calib'
    dataset_name = 'mot'
    seq_name     = 'MOT17-04-FRCNN'

    # calib_method = 'joint'

    # calib_method = 'iter'

    # calib_method = 'exh'

    # calib_method = 'de'

    calib_method = 'de_fov'

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

    if use_gt:
        track_file = os.path.join(root_dir, 'datasets', dataset_name, 'train', seq_name, 'gt/gt.txt')
    else:
        track_file = os.path.join(root_dir, 'results', proj_name, dataset_name)

    # boxloader    = BoxDataLoader(track_file = track_file,
    #                             img_size    = (w_img, h_img),
    #                             window      = int(seq_info['fps'] * 2),
    #                             stride      = seq_info['fps'] // 3,
    #                             height_dif_thresh = 3,
    #                             fps         = seq_info['fps'])

    boxloader    = BoxDataLoader(track_file = track_file,
                                 img_size    = (w_img, h_img),
                                 window      = int(seq_info['fps'] * 2),
                                 stride      = seq_info['fps'] // 2,
                                 height_dif_thresh  = 3,
                                 front_ratio_thresh = 0.8,
                                 fps         = seq_info['fps'])
    # boxloader    = BoxDataLoader(track_file = track_file,
    #                              img_size    = (w_img, h_img),
    #                              window      = int(seq_info['fps'] * 1.5),
    #                              stride      = seq_info['fps'] // 3,
    #                              height_dif_thresh  = 3,
    #                              front_ratio_thresh = 0.8,
    #                              fps         = seq_info['fps'])

    boxes       = boxloader.get_all_pairs() # N x 2 x 4

    print(f'samples: {boxes.shape}')

    cam_params = {'w_img': w_img, 'h_img': h_img}

    init_fov    = get_inital_fov(seq_name)

    init_fov = 80

    initials = [70, 0, init_fov]
    start_time = time.time()
    if calib_method == 'joint':
        result, loss_val = joint_calib(boxes, initials, cam_params, disp = True)
    elif calib_method == 'iter':
        result, loss_val = iteractive_calib(boxes, initials, cam_params, num_inters = 3, disp = False)
    elif calib_method == 'exh':
        result, loss_val = exhausive_calib(boxes, initials, cam_params, disp = False)
    elif calib_method == 'de':
        initials = ((20, 100), (-10, 10), (30, 120))
        result, loss_val = de_calib(boxes, initials, cam_params, disp = False)
    elif calib_method == 'de_fov':
        initials = ((20, 100), (-10, 10), (30, 120))
        result, loss_val = de_calib_v2(boxes, initials, cam_params, disp = False)

    end_time   = time.time()
    duration = end_time - start_time
    print(f'duration: {round(duration, 2)}s')

    alpha, beta, fov = result
    alpha, beta, fov = float(alpha), float(beta), float(fov)

    print(f'alpha: {alpha}, beta: {beta}, fov: {fov}(s: {fov_to_s(fov)}) loss:{round(loss_val, 4)} dur: {round(duration, 2)} s')

    # alpha, beta, fov = 90.13, -1.82, 30
    # alpha, beta, fov = 91, -2, 103.55
    # alpha, beta, fov = 59.66, 1.450, 55.1
    print(f'alpha {alpha}, beta {beta}, fov {fov}')

    # alpha, beta, fov = 91, -2, 103
    focal_length  = gen_focal_length(w_img, fov)

    # alpha, beta, fov = 58.0, 1.0, 65.19

    # print(float(alpha), float(beta))

    # print(alpha, beta, focal_length)

    rotation = Rotation(np.array([alpha, beta, 0]), mode = 'ZYX')
    camera   = Camera(w_img = w_img, 
                      h_img = h_img, 
                      sx = focal_length, 
                      sy = focal_length, 
                      cx = w_img / 2, 
                      cy = h_img / 2)
 
    aview    = AerialView(camera, rotation)

    # visualize calibration result
    track_dir     = os.path.join(root_dir, 'results', proj_name, dataset_name, seq_name)
    res_dir       = track_dir + '/ground_floor'
    gen_dir(res_dir)
    # all_tracks    = np.load(track_dir + '/tracks.npy')
    # all_track_ids = np.load(track_dir + '/track_ids.npy')

    all_tracks = boxloader.tracks
    all_track_ids = boxloader.track_ids

    frame_id   = 150
    num_frames = 100
    img_size   = 500 if not seq_name == 'MOT17-05-FRCNN' else 200
    # video_path = track_dir + f'/{seq_name}_{alpha.round(2)}_{beta.round(2)}_{fov[0].round(2)}.mp4'
    video_path = track_dir + f'/{seq_name}_{round(alpha, 2)}_{round(beta, 2)}_{round(fov, 2)}_{round(loss_val, 3)}_{calib_method}.mp4'
    print(video_path)
    fps        = 15
    output_vid = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_size + w_img, h_img))

    # for frame_id in range(num_frames):
    subset    = 'train' if seq_name in MOT17_TRAIN_SEQ_NAMES else 'test'
    img_dir   = os.path.join(root_dir, 'datasets', dataset_name, subset, seq_name, 'img1')
    img_file  = img_dir + '/{:06d}.jpg'.format(frame_id + 1)
    frame     = cv2.imread(img_file)
    # print(frame.shape, tracks.shape, track_ids.shape)
    
    cam_info = {'w_img': w_img, 'h_img': h_img, 'fov': fov,
                'alpha': alpha, 'beta': beta}

    # boxes = tracks[:, frame_id, :]
    # # xyh   = compute_xyh_from_boxes(cam_info, boxes)

    # # rngs = get_val_rngs(cam_info, tracks)
    # # print(f'xyh range: {rngs}')

    boxes        = all_tracks[:, frame_id, :]
    box_mask     = boxes[:, 2] * boxes[:, 3] > 0
    # # print(boxes.shape)
    boxes        = boxes[box_mask, :]
    # print(boxes.shape)
    # print(boxes.shape)
    track_ids    = all_track_ids[box_mask]
    # print(boxes.shape)
    bot_centers  = get_box_pts(boxes)

    draw_img = frame.copy()
    for i in range(bot_centers.shape[0]):
        track_id = track_ids[i]
        draw_text(draw_img, f'frame_id: {frame_id}', (5, 20), color = (0, 255, 0), scale = 1, thickness = 2)
        draw_corners(draw_img, bot_centers[i, :].reshape((-1, 2)), color = RANDOM_COLORS[track_id].tolist(), dot_size = 3)
        draw_box(draw_img, boxes[i, :], color = RANDOM_COLORS[track_id].tolist(), thickness = 2)

    # if bot_centers.shape[0] == 0: continue

    xy          = aview.map_img_to_grf(bot_centers, axis = 2)
    h           = aview.compute_heights(boxes).reshape((-1, 1))
    # # print(h)
    xyh          = np.concatenate([xy, h], axis = -1)
    xyh2         = compute_xyh_from_boxes(cam_info, boxes)

    # print(xyh[:10, :])
    # print(xyh2[:10, :])

    xyh_copy     = xyh.copy()
    # print(xyh.shape)
    # print(xyh_0[:10, 2].flatten(), xyh[:10, 2].flatten())

    xmin, xmax = xyh[:, 0].min(), xyh[:, 0].max()
    ymin, ymax = xyh[:, 1].min(), xyh[:, 1].max()
    # print(xyh[:, 0].mean(), xyh[:, 0].std(), 
    #       xyh[:, 1].mean(), xyh[:, 1].std(),
    #       xyh[:, 2].mean(), xyh[:, 2].std())
    # print(xmin, xmax, ymin, ymax)
    # rngs = [max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax))]
    rngs = [xmin, ymin, xmax, ymax]

    vis_3d_draw = vis_xyz_as_images(xyh_copy, track_ids, rngs, img_size = img_size, dot_size = 5, 
                        scale = 1, thickness = 2, orient_angle = 0, r = 10, 
                        arrow_thickness = 2, colors = RANDOM_COLORS)
    # vis_3d_draw = plot_2d_pts_as_imgs(xyh, val_rng = [20, 100], labels = ['x', 'y'], 
    #                     fig_size = 5, title = '')
    
    blk_board   = create_black_board(img_size, h_img)
    # print(blk_board.shape, vis_3d_draw.shape)
    blk_board[h_img // 2 - img_size // 2: h_img // 2 + img_size // 2, :] = vis_3d_draw
    # cv2.imwrite(track_dir + f'/vis_3d_frame-{frame_id}.png', vis_3d_draw)
    # cv2.imwrite(track_dir + f'/vis_2d_frame-{frame_id}.png', draw_img)
    # board = create__board(w_img = img_size, )
    # print(draw_img.shape, blk_board.shape)
    vis_concat = cv2.hconcat([draw_img, blk_board])
    cv2.imwrite(res_dir + '/ground_floor.png', vis_concat)
        # output_vid.write(vis_concat)

    # project back to image plane
    # print(xyh.shape)
    top_xyh = xyh.copy()
    top_xyh[:, 2] = 1 - top_xyh[:, 2]

    bot_xyh = xyh.copy()
    bot_xyh[:, 2] = 1.

    # print(bot_xyh)

    xyhs    = np.concatenate([top_xyh, bot_xyh], axis = 0)
    rays    = rotation.rotate_inverse(xyhs.T)
    img_pts = camera.map_3d_rays_to_2d_pts(rays)

    img_pts = np.int16(img_pts)

    top_pts = img_pts[:len(top_xyh), :]
    bot_pts = img_pts[len(top_xyh):, :]

    draw_img2 = draw_img.copy()
    thickness = 2
    for i in range(top_pts.shape[0]):
        top_pt = top_pts[i, :]
        bot_pt = bot_pts[i, :]
        track_id = track_ids[i]
        color = (0, 0, 255)
        cv2.line(draw_img2, top_pt, bot_pt, color, thickness)

    concat2 = cv2.hconcat([draw_img2, blk_board])
    cv2.imwrite(res_dir + f'/image_pts_{alpha}_{beta}_{fov}.png', draw_img2)
    cv2.imwrite(res_dir + f'/image_pts_{round(alpha, 2)}_{round(beta, 2)}_{round(fov, 2)}_v2.png', concat2)

if __name__ == '__main__':
    main()