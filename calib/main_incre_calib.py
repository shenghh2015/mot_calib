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
import io


import matplotlib.pyplot as plt

from visualize import create_black_board, draw_text, vis_xyz_as_images, vis_box_pairs, plot_height_stats, create_bordered_white_board
from datasets_init import BoxDataLoader
from utils import load_seq_info, save_to_json


from const import MOT17_SEQ_NAMES, MOT17_TRAIN_SEQ_NAMES, MOT17_STATIC_SEQ_NAMES
from yolox.camera.img_vis import draw_box, draw_corners, draw_boxes, draw_text

import time

from cardboard.new_cardboard_v4 import CardBoard
from cardboard.camera_utils import Camera as Camera2
from cardboard.geometry_3d import Rotation as Rotation2
from cardboard.incre_calib_v3 import IncreCalib

# init ECC module
number_of_iterations = 50
termination_eps = 0.01
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

def convert_pyplot_to_image(fig, dpi=180):
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

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

def gen_box_mask(boxes, w_img = 1920, h_img = 1080):

    box_mask = np.zeros((h_img, w_img), dtype = np.uint8)

    boxes_copy = np.int16(boxes.copy())
    for i in range(boxes.shape[0]):
        box = boxes_copy[i, :]
        tl  = box[:2]
        br  = tl + box[2:]
        box_mask[tl[0]: br[0], tl[1]: br[1]] = 1

    return box_mask

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

def load_calib_info(file_name):
    import json
    json_data = json.load(open(file_name))
    return json_data

def gradient(gray):
    gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
    gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude

def main():

    args     = get_params()
    root_dir = os.path.abspath('../')
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

    calib_res_dir = os.path.join(root_dir, 'incr_calib_results', dataset_name)
    gen_dir(calib_res_dir)

    vis_res_dir  = os.path.join(root_dir, 'incr_calib_results', dataset_name, seq_name, 'vis_results')
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

    frame_window   = time_window * seq_info['fps']
    end_frame_id   = start_frame_id + frame_window
    ref_frame_id   = start_frame_id
    boxes          = boxloader.get_ecc_warpped_boxes_v2(start_frame_id, frame_window, 
                                                        ref_frame_id = 0, use_ecc = use_ecc)
    print(f'samples: {tuple(boxes.shape)}')

    calib_info_file = calib_res_dir + f'/{seq_name}.json'

    if not os.path.exists(calib_info_file):
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
    else:
        calib_info = load_calib_info(calib_info_file)
        print(calib_info)
    
    # generate a mapping
    focal_length = gen_focal_length(calib_info['w_img'], calib_info['fov'])
    # camera2    = Camera2(focal_length, focal_length, calib_info['w_img']/2, calib_info['h_img'] / 2)
    camera2 = Camera2(w_img = calib_info['w_img'], 
                       h_img = calib_info['h_img'], 
                       sx = focal_length, 
                       sy = focal_length,
                       cx = calib_info['w_img']/2, 
                       cy = calib_info['h_img'] / 2)
    rotation2  = Rotation2(np.array([calib_info['alpha'], calib_info['beta'], 0]), mode = 'ZYX')
    # print('check point 0')
    # print(camera2.Mint)
    cb_mapping = CardBoard(camera2, rotation2)
    inc_calib  = IncreCalib(calib_info['alpha'], calib_info['beta'], calib_info['fov'])

    num_frames = 650

    res_fps      = 10
    res_w, res_h = w_img // 4, h_img // 4 + 360    # 540, 270
    vid_writer = cv2.VideoWriter(
        vis_res_dir + '/calib_seq.mp4', cv2.VideoWriter_fourcc(*"mp4v"), res_fps, (res_w, res_h)
    )

    alpha_list, beta_list, fov_list = [], [], []
    dx_list, dy_list, dz_list = [], [], []
    
    for frame_id in range(ref_frame_id, ref_frame_id + num_frames):
        
        print(f'----------------frame id: {frame_id + 1}----------')
        # refrence frame (t - 1)
        ref_boxes  = boxloader.raw_tracks[:, frame_id, :].copy()
        ref_boxes  = ref_boxes[ref_boxes[:, -1] > 0, :]
        ref_frame  = cv2.imread(boxloader.img_dir + '/{:06d}.jpg'.format(frame_id + 1))
        box_mask   = gen_box_mask(ref_boxes, w_img = 1920, h_img = 1080)

        # current frame (t)
        obs_boxes  = boxloader.raw_tracks[:, frame_id + 1, :].copy()
        obs_boxes  = obs_boxes[obs_boxes[:, -1] > 0, :]
        frame      = cv2.imread(boxloader.img_dir + '/{:06d}.jpg'.format(frame_id + 2))

        # frame  = ref_frame.copy()
        print('**** Use pixel values:')
        warp_matrix = inc_calib.compute_ecc(frame, ref_frame, dsample = 4, 
                                            box_mask = box_mask, use_mask = False)
        print(warp_matrix)

        # print(warp_matrix)
        # print(m_warp_matrix)
        dsample = 4
        h, w    = frame.shape[:2]
        h, w    = h // dsample, w // dsample
        frame      = cv2.resize(frame,     (w, h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame_gray.shape, frame_gray.max(), frame_gray.min())

        ref        = cv2.resize(ref_frame, (w, h))
        ref_gray   = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        print(ref_gray.shape, ref_gray.max(), ref_gray.min())

        warp_frame = cv2.warpPerspective(frame_gray, warp_matrix, (w, h))

        # cv2.imwrite(vis_res_dir + f'/ref_{frame_id}.png', ref_gray)
        # cv2.imwrite(vis_res_dir + f'/warp_{frame_id}.png', warp_frame)

        diff1 = np.abs(frame_gray.astype(np.float32) - ref_gray.astype(np.float32))
        diff2 = np.abs(warp_frame.astype(np.float32) - ref_gray.astype(np.float32))
        print(diff1.max(), diff1.min())
        # cv2.imwrite(vis_res_dir + f'/dif1_{frame_id}.png', diff1.astype(np.uint8))
        # cv2.imwrite(vis_res_dir + f'/dif2_{frame_id}.png', diff2.astype(np.uint8))

        ecc1 = cv2.computeECC(templateImage=frame_gray, inputImage=ref_gray)
        ecc2 = cv2.computeECC(templateImage=warp_frame, inputImage=ref_gray)

        print(f'ECC score: {round(ecc1, 4)}')
        print(f'ECC score: {round(ecc2, 4)}')

        # print('**** Use gradients:')
        # # use graidents
        # g_frame = gradient(frame_gray)
        # g_ref   = gradient(ref_gray)
        # warp_matrix2 = np.eye(3,3, dtype = np.float32)
        # (cc, warp_matrix2) = cv2.findTransformECC(g_frame, g_ref, warp_matrix2, cv2.MOTION_HOMOGRAPHY, criteria)

        # print(warp_matrix2)
        # g_warp  = cv2.warpPerspective(g_frame, warp_matrix2, (w, h))
        # ecc1 = cv2.computeECC(templateImage=g_frame, inputImage=g_ref)
        # ecc2 = cv2.computeECC(templateImage=g_warp, inputImage=g_ref)

        # print(f'ECC score: {round(ecc1, 4)}')
        # print(f'ECC score: {round(ecc2, 4)}')

        # g_diff1 = np.abs(g_frame.astype(np.float32) - g_ref.astype(np.float32))
        # g_diff2 = np.abs(g_warp.astype(np.float32) - g_ref.astype(np.float32))
        # print(g_diff2.max(), g_diff2.min())
        # cv2.imwrite(vis_res_dir + f'/g_ref_{frame_id}.png',  g_ref)
        # cv2.imwrite(vis_res_dir + f'/g_warp_{frame_id}.png', g_warp)
        # cv2.imwrite(vis_res_dir + f'/g_dif1_{frame_id}.png', g_diff1.astype(np.uint8))
        # cv2.imwrite(vis_res_dir + f'/g_dif2_{frame_id}.png', g_diff2.astype(np.uint8))

        # print('before the calibration 0: ', cb_mapping.rotation.get_angles())
        # print(cb_mapping.camera.Mint)
        # print(cb_mapping.transition)

        track_boxes = inc_calib.map_to_ref(warp_matrix, obs_boxes, dsample = 4)
        xyhw        = cb_mapping.bbox2xyhw(track_boxes)
        # print(track_boxes[:2,:])
        # print(obs_boxes[:2,:])
        # print('before the calibration: ', cb_mapping.rotation.get_angles())
        # print(cb_mapping.camera.Mint)
        # print(cb_mapping.rotation.get_R())
        # print(cb_mapping.transition)
        # new_mapping, new_T = inc_calib.update_calib_GD_v4(obs_boxes, track_boxes, cb_mapping, num_iters = 5)

        # visualization
        draw_board1 = create_bordered_white_board(1920, 1080)
        draw_boxes(draw_board1, obs_boxes, (0, 0, 255), thickness = 2)
        projections = cb_mapping.xyhw2bbox(xyhw)
        draw_boxes(draw_board1, projections, (0, 255, 255), thickness = 2)

        new_mapping, new_T = inc_calib.update_calib_GD_v3(obs_boxes, track_boxes, cb_mapping, num_iters = 50)

        # new_mapping, new_T = inc_calib.update_calib_GD_v2(obs_boxes, track_boxes, cb_mapping, num_iters = 50)
        
        # box_centers   = obs_boxes[:, :2] + obs_boxes[:, 2:] / 2
        # track_centers = track_boxes[:, :2] + track_boxes[:, 2:] / 2
        # initial_loss  = np.linalg.norm(box_centers - track_centers, axis = 1).mean()
        # print(f'test initial loss: {round(initial_loss, 3)}')

        # print('After the calibration: ', cb_mapping.rotation.get_angles())
        # print(cb_mapping.camera.Mint)
        # print(cb_mapping.transition)
        # new_mapping, _     = inc_calib.update_calib_GD(obs_boxes, track_boxes, cb_mapping, num_iters = 5)
        cb_mapping    = inc_calib.update_cam_states(cb_mapping)
        # new_mapping = cb_mapping

        # after 
        new_boxes   = new_mapping.xyhw2bbox(xyhw)
        draw_board2 = create_bordered_white_board(1920, 1080)
        draw_boxes(draw_board2, obs_boxes, (0, 0, 255), thickness = 2)
        draw_boxes(draw_board2, new_boxes, (0, 255, 255), thickness = 2)

        # concat 
        draw_concat = cv2.vconcat([draw_board1, draw_board2])

        if frame_id < 10:
            cv2.imwrite(vis_res_dir + f'/calib_{frame_id}.png', draw_concat)

        # draw calibration results
        alpha_list.append(inc_calib.alpha)
        beta_list.append(inc_calib.beta)
        fov_list.append(inc_calib.fov)
        dx_list.append(new_T[0])
        dy_list.append(new_T[1])
        dz_list.append(new_T[2])
        fig = plt.figure(figsize=(4, 1.5), dpi=120)
        fig.tight_layout()
        plt.plot(alpha_list, color='b')
        plt.plot(beta_list, color='g')
        plt.plot(fov_list, color='r')
        plt.legend(['alpha', 'beta', 'fov'])

        plt.xlim([0, num_frames])
        plt.ylim([-5, 90])
        plt.close(fig)

        img = convert_pyplot_to_image(fig, dpi=120)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)

        fig = plt.figure(figsize=(4, 1.5), dpi=120)
        fig.tight_layout()
        plt.plot(dx_list, color='b')
        plt.plot(dy_list, color='g')
        plt.plot(dz_list, color='r')
        plt.legend(['dx', 'dy', 'dz'])

        plt.xlim([0, num_frames])
        # plt.ylim([0, 90])
        plt.close(fig)

        img2 = convert_pyplot_to_image(fig, dpi=120)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        d_frame = cv2.resize(frame, (w_img // 4, h_img // 4))
        draw_text(d_frame, f'frame {frame_id}', (5, 40), (0, 255, 0), scale = 1, thickness = 3)
        draw_concat2 = cv2.vconcat([d_frame, img])
        draw_concat2 = cv2.vconcat([draw_concat2, img2])
        vid_writer.write(draw_concat2)

if __name__ == '__main__':
    main()