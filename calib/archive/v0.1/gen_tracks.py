import os
import cv2
from yolox.camera.vis import draw_box, draw_text
import random
import numpy as np
from utils import get_front_bboxes

from const import MOT17_TRAIN_SEQ_NAMES, MOT17_SEQ_NAMES

random.seed(0)
nb_color = 10000
colors = np.random.randint(0,255,(nb_color,3))

def load_tracks(tracking_file):
    boxes = {}
    with open(tracking_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(',')
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
    
    return track_numpy, track_ids_to_indices, frame_ids_to_indices, frame_ids, track_ids

def gen_dir(folder):
    if not os.path.exists(folder): 
        os.system(f'mkdir -p {folder}')

def main():

    root_dir     = os.path.abspath('../')
    dataset      = 'mot'
    # res_name     = 'mot17_paper_v3/yolox_x_ablation/bytetrack/'
    res_name     = 'mot17_paper_v3/bytetrack/'
    
    proj_name    = 'calib'

    # seq_name     = 'MOT17-04-FRCNN'
    # for seq_name in MOT17_TRAIN_SEQ_NAMES:
    for seq_name in MOT17_SEQ_NAMES:

        print(f'process {seq_name}')

        tracking_dir = os.path.join(root_dir, 'results', res_name)
        vis_dir      = os.path.join(root_dir, f'results/{proj_name}/{dataset}/{seq_name}')
        gen_dir(vis_dir)

        tracking_file = tracking_dir + f'/{seq_name}.txt'
        numpy_tracks, _, _, _, track_ids = load_tracks_from_tracking(tracking_file)
        print(numpy_tracks.shape, len(track_ids))
        
        np.save(vis_dir + '/tracks.npy', numpy_tracks)
        np.save(vis_dir + '/track_ids.npy', np.array(track_ids))

if __name__ == '__main__':
    main()