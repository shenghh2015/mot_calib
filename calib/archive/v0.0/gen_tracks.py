import cv2 as cv
import numpy as np
import os
import random
import json

import vis
import utils
import sys
import argparse

from yolox.camera import img_vis

sys.path.insert(0, os.path.abspath(".."))

img_vis

random.seed(0)
nb_color = 10000
colors = np.random.randint(0,255,(nb_color,3))
ROOT_DIR = os.path.abspath('../')

def get_all_boxes(file_name):
    boxes = {}
    with open(file_name, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
            # print(splits)
            frame_id = int(float(splits[0]))
            if not frame_id in boxes:
                boxes[frame_id] = {}
                boxes[frame_id]['ids'] = []
                boxes[frame_id]['box'] = []
            box_id = int(float(splits[1]))
            box = [float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5])]
            boxes[frame_id]['ids'].append(box_id)
            boxes[frame_id]['box'].append(box)
    return boxes

def get_all_tracks(file_name):
    tracks = {}
    with open(file_name, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
            track_id = int(float(splits[1]))
            frame_id = int(float(splits[0]))
            box = [float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5])]
            if not track_id in tracks:
                tracks[track_id] = {}
                tracks[track_id]['frame_id'] = []
                tracks[track_id]['box'] = []
            tracks[track_id]['frame_id'].append(frame_id)
            tracks[track_id]['box'].append(box)
    return tracks

def create_draw_board(w_img, h_img):
    return np.zeros((h_img, w_img, 3), dtype = np.uint8)

sequences = {
    'static': ['MOT17-09', 'MOT17-04', 'MOT17-02'],
    'dynamic': ['MOT17-13', 'MOT17-11', 'MOT17-10', 'MOT17-05'],
}

test_sequences = {
    'static': ['MOT17-01', 'MOT17-03', 'MOT17-08'],
    'dynamic': ['MOT17-06', 'MOT17-07', 'MOT17-12', 'MOT17-14'],  
}

seq_fps_dic = {
    'MOT17-13': 25,
    'MOT17-11': 30,
    'MOT17-10': 30,
    'MOT17-09': 30,
    'MOT17-05': 14,
    'MOT17-04': 30,
    'MOT17-02': 30}

test_seq_fps_dic = {
    'MOT17-01': 30,
    'MOT17-03': 30,
    'MOT17-08': 30,
    'MOT17-06': 14,
    'MOT17-07': 30,
    'MOT17-12': 30,
    'MOT17-14': 25}

def parse_info(info_file_name):
    info = {}
    with open(info_file_name, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            seps = line.strip().split('=')
            if seps[0] == 'seqLength':
                info['seq_len'] = int(seps[1])
            if seps[0] == 'frameRate':
                info['fps'] = int(seps[1])
            if seps[0] == 'imWidth':
                info['w_img'] = int(seps[1])
            if seps[0] == 'imHeight':
                info['h_img'] = int(seps[1])
    return info

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_group", type=str, default = 'static')
    parser.add_argument("--seq_index", type=int, default = 0)           
    args = parser.parse_args()
    return args

def load_tracks_from_tracking(tracking_file):
    frame_ids = []
    track_ids = []
    with open(tracking_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
            frame_ids.append(int(splits[0]))
            track_ids.append(int(splits[1]))

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
        frame_id, track_id = int(splits[0]), int(splits[1])
        box  = np.array([float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5])])
        track_numpy[track_ids_to_indices[track_id], frame_ids_to_indices[frame_id], :] = box
    
    # print(track_numpy)
    return track_numpy, track_ids_to_indices, frame_ids_to_indices, frame_ids

if __name__ == '__main__':

    args = get_params()

    data_dir = ROOT_DIR + '/data/'
    dataset = 'mot17'
    METHOD_FLAG = 'FRCNN'

    # group = 'static'
    # seq_index = 0
    group     = args.seq_group
    seq_index = args.seq_index
    seq_id = test_sequences[group][seq_index]
    seq_name = seq_id + '-' + METHOD_FLAG

    # get seq info
    test_data_dir = data_dir + '/' + dataset.upper() + '/test/'
    info_file_name = test_data_dir + '/{}'.format(seq_name) + '/seqinfo.ini'
    info = parse_info(info_file_name)

    # load tracking results
    tracking_txt_file = test_data_dir + '/yolox_det_v2/{}.txt'.format(seq_name)
    numpy_tracks, track_table, frame_ids_to_indices, frame_ids = load_tracks_from_tracking(tracking_txt_file)
    image_dir = test_data_dir + '/{}/img1'.format(seq_name)

    frame_names = [fn for fn in os.listdir(image_dir) if fn.endswith('.jpg')]
    frame_names.sort()

    # save track in numpy format
    track_dir = data_dir + '/' + dataset + '/{}'.format(seq_id)
    track_data_dir =  track_dir + '/numpy_tracks'
    utils.gen_dir(track_data_dir)
    track_vis_dir  = track_dir + '/numpy_vis'
    utils.gen_dir(track_vis_dir)
    res_dir = track_dir + '/results/'
    utils.gen_dir(res_dir)

    fps = info['fps']
    w_img, h_img = info['w_img'], info['h_img']
    out_video = cv.VideoWriter(res_dir + '/{}_boxes.avi'.format(seq_id),
                cv.VideoWriter_fourcc('M','J','P','G'), 
                fps, (w_img, h_img))

    original_video = cv.VideoWriter(res_dir + '/{}.avi'.format(seq_id),
                cv.VideoWriter_fourcc('M','J','P','G'), 
                fps, (w_img, h_img))

    board = img_vis.create_draw_board(w_img = w_img, h_img = h_img)
    board_draw = board.copy()

    w_img, h_img = info['w_img'], info['h_img']
    
    # visualize the tracks in a video and collect track for calibration
    for img_id, img_name in enumerate(frame_names):
        frame_name = frame_names[img_id]
        frame_id = int(frame_name.split('.')[0])
        if not frame_id in frame_ids_to_indices: continue
        img_index = frame_ids_to_indices[frame_id]
        img_file = image_dir + '/{}'.format(img_name)
        image = cv.imread(img_file)
        original_video.write(image)
        for trk_id in range(numpy_tracks.shape[0]):
            # box = numpy_tracks[trk_id, img_id, :]
            box = numpy_tracks[trk_id, img_index, :]
            x, y, w, h = box.flatten()
            # print(x, y, w, h)
            if x >=0 and x + w < w_img and y >=0 and y + h < h_img:
                # box = np.array(trk['bbox'])
                # numpy_tracks[trk_id_np, img_id_np, :] = box
                vis.draw_box(image, box, colors[trk_id].tolist(), thickness = 2)
                pt = box[:2] + box[2:] /2 
                vis.draw_text(image, '{}'.format(trk_id), pt,
                    color = colors[trk_id].tolist(),
                    scale = 2, thickness = 2)
                top_pt = box[:2]
                top_pt[0] += box[2] / 2
                box_size = w * h 
                vis.draw_text(image, '{}'.format(int(box_size)), top_pt,
                    color = colors[trk_id].tolist(),
                    scale = 2, thickness = 2)
        out_video.write(image)
    out_video.release()
    original_video.release()

    # save numpy tracks
    np.save(track_data_dir + '/tracks.npy', numpy_tracks)
