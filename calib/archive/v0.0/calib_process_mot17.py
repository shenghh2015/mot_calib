from calib.const import RANDOM_COLORS
from camera_utils import Camera, AerialView
from calib.geometry import compute_xyh_from_boxes, gen_focal_length
from calib.geometry_3d import Rotation
from calib_losses import alpha_beta_loss
import numpy as np
import os
import cv2
from scipy import optimize

from visualize import vis_xyz_as_images, plot_2d_pts_as_imgs
from datasets import BoxDataLoader
from utils import get_box_centers, load_seq_info

from const import MOT17_SEQ_NAMES, MOT17_TRAIN_SEQ_NAMES, MOT17_STATIC_SEQ_NAMES
from yolox.camera.img_vis import draw_corners

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

def main():

    root_dir = os.path.abspath('../')
    proj_name    = 'calib'
    dataset_name = 'mot'
    seq_name     = 'MOT17-04-FRCNN'

    seq_set = MOT17_STATIC_SEQ_NAMES

    for seq_name in seq_set:

        # if seq_name == 'MOT17-05-FRCNN' or seq_name == 'MOT17-09-FRCNN' \
        #     or seq_name == 'MOT17-10-FRCNN' or seq_name == 'MOT17-11-FRCNN' \
        #         or seq_name == 'MOT17-13-FRCNN': continue

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
        fov          = 65

        dataset_dir = os.path.join(root_dir, 'results', proj_name, dataset_name)
        boxloader   = BoxDataLoader(dataset_dir, seq_name,
                                    window      = int(seq_info['fps'] * 2),
                                    interval    = 10,
                                    stride      = seq_info['fps'] // 2,
                                    trj_thres   = 0.8,
                                    h_dif_thres = 2,
                                    fps         = seq_info['fps'])
        boxes       = boxloader.get_all_hpairs() # N x 4 x 4

        print(f'samples: {boxes.shape}')

        init_angles = [70, 0]
        print(f'inital angles: {init_angles}')

        cam_params = {'w_img': w_img, 'h_img': h_img, 'fov': fov}
        print(f'camera info: {cam_params}')

        params     = (cam_params, boxes)
        result     = optimize.minimize(alpha_beta_loss, init_angles, args = params, 
                    method = 'BFGS', options={'gtol': 1e-10, 'disp': True})
        print(f'Optimized angles: {np.round(result.x, 2)}, loss {np.round(result.fun, 3)}')
        # alpha, beta = result.x

        # visualize calibration result
        # frame_id = 2
        # subset    = 'train' if seq_name in MOT17_TRAIN_SEQ_NAMES else 'test'
        # img_dir   = os.path.join(root_dir, 'datasets', dataset_name, subset, seq_name, 'img1')
        # img_file  = img_dir + '/{:06d}.jpg'.format(frame_id)
        # print(img_file)
        # frame     = cv2.imread(img_file)
        # track_dir = os.path.join(root_dir, 'results', proj_name, dataset_name, seq_name)
        # tracks    = np.load(track_dir + '/tracks.npy')
        # track_ids = np.load(track_dir + '/track_ids.npy')
        # print(frame.shape, tracks.shape, track_ids.shape)
        
        # cam_info = {'w_img': w_img, 'h_img': h_img, 'fov': fov,
        #             'alpha': alpha, 'beta': beta}

        # boxes = tracks[:, frame_id, :]
        # # xyh   = compute_xyh_from_boxes(cam_info, boxes)

        # # rngs = get_val_rngs(cam_info, tracks)
        # # print(f'xyh range: {rngs}')

        # focal_length = gen_focal_length(w_img, fov)

        # print(alpha, beta, focal_length)
        # rotation = Rotation(np.array([alpha, beta, 0]))
        # camera   = Camera(w_img = 1920, 
        #                 h_img = 1080, 
        #                 sx = focal_length, 
        #                 sy = focal_length, 
        #                 cx = w_img / 2, 
        #                 cy = h_img / 2)

        # aview   = AerialView(camera, rotation)

        # box_mask = boxes[:, 2] * boxes[:, 3] > 0
        # # print(boxes.shape)
        # boxes    = boxes[box_mask, :]
        # track_ids    = track_ids[box_mask]
        # print(boxes.shape)
        # bot_centers = get_box_pts(boxes)

        # draw_img = frame.copy()
        # for i in range(bot_centers.shape[0]):
        #     track_id = track_ids[i]
        #     draw_corners(draw_img, bot_centers[i, :].reshape((-1, 2)), color = RANDOM_COLORS[track_id].tolist(), dot_size = 3)

        # xyh_0   = compute_xyh_from_boxes(cam_info, boxes)

        # xy          = aview.map_img_to_grf(bot_centers, axis = 2)
        # h           = aview.compute_heights(boxes).reshape((-1, 1))
        # # print(h)
        # xyh         = np.concatenate([xy, h], axis = -1)
        # # print(xyh)
        # print(xyh_0[:10, 2].flatten(), xyh[:10, 2].flatten())

        # xmin, xmax = xyh[:, 0].min(), xyh[:, 0].max()
        # ymin, ymax = xyh[:, 1].min(), xyh[:, 1].max()
        # print(xyh[:, 0].mean(), xyh[:, 0].std(), xyh[:, 1].mean(), xyh[:, 1].std())
        # rngs = [max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax))]

        # # vis_3d_draw = vis_xyz_as_images(xyh, track_ids, rngs, img_size = 500, dot_size = 2, 
        # #                  scale = 1, thickness = 2, orient_angle = 90, r = 10, arrow_thickness = 2)
        # vis_3d_draw = plot_2d_pts_as_imgs(xyh, val_rng = [20, 100], labels = ['x', 'y'], 
        #                     fig_size = 5, title = '')

        # cv2.imwrite(track_dir + f'/vis_3d_frame-{frame_id}.png', vis_3d_draw)
        # cv2.imwrite(track_dir + f'/vis_2d_frame-{frame_id}.png', draw_img)

if __name__ == '__main__':
    main()