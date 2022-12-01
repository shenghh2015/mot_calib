import os
import json
import sys 
sys.path.append("..")
import cv2
import numpy as np
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

from const.colors import get_random_colors
COLORS = get_random_colors()

from tools.visualize import (draw_box, draw_text, create_blk_board)

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system(f'mkdir -p {folder}')

def load_gt(gt_file):
    json_data = json.load(open(gt_file))    
    return json_data

def load_seq_info(seq_info_file):
    info = {}
    with open(seq_info_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            if '=' in line:
                splits = line.strip().split('=')
                info[splits[0]] = splits[1]

    return info

DATA_DIR = '/media/workspace/yirui/data/tracking_reid/prepared/MOTSynth/'

def main():

    for seq_id in [0, 1, 3, 4, 5, 6]:
        # seq_id   = 13
        img_dir  = DATA_DIR + '/MOTSynth_train/frames/{:03d}'.format(seq_id)
        vid_file = DATA_DIR + '/MOTSynth_1/{:03d}.mp4'.format(seq_id)
        gt_file  = DATA_DIR + '/annotations/{:03d}.json'.format(seq_id)
        seq_info_file = DATA_DIR + '/mot_annotations/{:03d}/seqinfo.ini'.format(seq_id)

        gts      = load_gt(gt_file)
        cam_info = gts['images'][0]
        print(cam_info)

        in_video   = cv2.VideoCapture(vid_file)
        img_w      = int(in_video.get(CAP_PROP_FRAME_WIDTH))
        img_h      = int(in_video.get(CAP_PROP_FRAME_HEIGHT))
        fps        = in_video.get(CAP_PROP_FPS)
        num_frames = int(in_video.get(CAP_PROP_FRAME_COUNT))
        
        down_fact    = 4
        w_out, h_out = img_w // down_fact, img_h // down_fact 
        res_dir    = os.path.abspath('../results/motsynth')
        gen_dir(res_dir)
        out_video  = cv2.VideoWriter(
            res_dir + '/{:03d}_cam_motion.mp4'.format(seq_id), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out + h_out * 3 // 2 , h_out)
        )
        
        # load seq info
        seq_info = load_seq_info(seq_info_file)

        # stop_frame_id = 10
        for i in range(num_frames):
            ret, frame = in_video.read()
            if not ret: continue

            board      = create_blk_board(h_out * 3 // 2 , h_out)
            img_info   = gts['images'][i]

            # extrinsic parameters
            draw_text(board, f'pos: {np.round(img_info["cam_world_pos"], 1)}', (5, 30), (0, 255, 0), scale = 0.5, thickness = 1)
            draw_text(board, f'rot: {np.round(img_info["cam_world_rot"], 2)}', (5, 60),(0, 255, 0), scale = 0.5, thickness = 1)

            # intrinsic parameters
            draw_text(board, f'FOV:{seq_info["FOV"]}, fx:{seq_info["fx"]}, fy:{seq_info["fy"]}', (5, 90), (0, 0, 255), scale = 0.5, thickness = 1)
            draw_text(board, f'cx:{seq_info["cx"]}, cy:{seq_info["cy"]}', (5, 120), (0, 0, 255), scale = 0.5, thickness = 1)

            d_frame    = cv2.resize(frame, (w_out, h_out))
            draw_board = cv2.hconcat([d_frame, board])
            out_video.write(draw_board)

if __name__ == '__main__':
    main()
