import os
from calib.utils import get_front_bboxes, get_front_bboxes_nms
import cv2
from yolox.camera.vis import draw_box, draw_text
import random
import numpy as np

MOT20_SEQ_NAMES = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-04', 
                   'MOT20-05', 'MOT20-06', 'MOT20-07', 'MOT20-08']

TRAIN_SEQ_NAMES = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

random.seed(0)
nb_color = 10000
colors = np.random.randint(0,255,(nb_color,3))

def load_tracks(tracking_file):
    boxes = {}
    with open(tracking_file, 'r+') as f:
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

def gen_dir(folder):
    if not os.path.exists(folder): 
        os.system(f'mkdir -p {folder}')

def main():

    root_dir     = os.path.abspath('../')
    dataset      = 'MOT20'
    res_name     = 'mot20_paper/yolox_x_mix_mot20_ch/bytetrack/'
    

    rule_front = 'nms'
    seq_name = 'MOT20-02'
    proj_name = 'calib'

    tracking_dir = os.path.join(root_dir, 'results', res_name)
    vis_dir      = os.path.join(root_dir, f'results/{proj_name}/{dataset}/{seq_name}')
    gen_dir(vis_dir)

    if seq_name in TRAIN_SEQ_NAMES:
        img_dir      = os.path.join(root_dir, 'datasets', dataset, 'train', f'{seq_name}/img1')
    else:
        img_dir      = os.path.join(root_dir, 'datasets', dataset, 'test', f'{seq_name}/img1')
    
    track_res = load_tracks(tracking_dir + f'/{seq_name}.txt')

    frame_id = 2
    img_file = os.path.join(img_dir, '{:06d}.jpg'.format(frame_id))
    img = cv2.imread(img_file)
    h_img, w_img = img.shape[:2]

    frame     = img.copy()
    boxes     = track_res[frame_id]['box']
    track_ids = track_res[frame_id]['ids']
    for i, box in enumerate(boxes):
        trk_id = track_ids[i]
        x, y, w, h = box
        scale = 0.8
        draw_box(frame, box, colors[trk_id].tolist(), thickness = 2)
        draw_text(frame, f'{trk_id}', (x + w - 5, y - 5), colors[trk_id].tolist(), scale = scale, thickness = 2)
    cv2.imwrite(vis_dir + f'/{seq_name}_{frame_id}.png', frame)

    bboxes      = np.array(boxes).reshape((-1, 4))
    front_inds  = get_front_bboxes(bboxes, h_img) if not rule_front == 'nms' else get_front_bboxes_nms(bboxes, h_img = h_img)

    print(front_inds)

    frame       = img.copy()
    front_boxes = bboxes[front_inds, :]
    front_ids   = np.array(track_ids)[front_inds]
    for ind in range(front_boxes.shape[0]):
        box    = front_boxes[ind]
        trk_id = front_ids[ind]
        x, y, w, h = box
        if trk_id == 96:
            print(y + h, h_img)
        scale = 0.8
        draw_box(frame, box, colors[trk_id].tolist(), thickness = 2)
        draw_text(frame, f'{trk_id}', (x + w - 5, y - 5), colors[trk_id].tolist(), scale = scale, thickness = 2)
    cv2.imwrite(vis_dir + f'/{seq_name}_{frame_id}_front_{rule_front}.png', frame)

if __name__ == '__main__':
    main()