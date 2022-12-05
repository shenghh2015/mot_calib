from calib.geometry import fov_to_s
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

from const import EPS

def get_front_bboxes(bboxes, h_img = 1080):

    # bboxes (tlwh): N x 4
    bboxes = np.array(bboxes).reshape((-1, 4))
    inds   = bboxes.shape[0]

    # bottom centers: N x 4
    bot_centers = bboxes[:, :2].copy()
    bot_centers[:, 0] += bboxes[:, 2] / 2   # x + w / 2
    bot_centers[:, 1] += bboxes[:, 3]       # y + h

    # tlbrs
    tlbrs        = bboxes.copy()
    tlbrs[:, 2:] = tlbrs[:, 2:] + tlbrs[:, :2]

    # compute y order: N x N
    front        = bot_centers[:, 1].reshape((-1, 1)) < bot_centers[:, 1].reshape((1, -1))
    iou_distance = bbox_ious(np.ascontiguousarray(tlbrs, dtype=np.float),
                             np.ascontiguousarray(tlbrs, dtype=np.float)
                            )

    cond_mask = np.logical_and(iou_distance > 0., front)
    cond_vect = cond_mask.sum(axis = -1)
    cond      = np.logical_and(cond_vect == 0., bot_centers[:, 1] <= h_img)
    inds      = np.where(cond)[0]

    return inds

def get_front_bboxes_nms(bboxes, h_img = 1080):

    bboxes   = np.array(bboxes).reshape((-1, 4))
    tlbrs    = bboxes.copy()
    tlbrs[:, 2:] += tlbrs[:, :2]

    box_area = bboxes[:, 2] * bboxes[:, 3]
    ious     = bbox_ious(np.ascontiguousarray(tlbrs, dtype=np.float),
                         np.ascontiguousarray(tlbrs, dtype=np.float)
                        )
    
    # remove boxes that are out of the image
    box_area[tlbrs[:, 3] >= h_img] = -1

    inds = []

    while True:
        max_ind = np.argmax(box_area)
        if box_area[max_ind] < 0: break
        inds.append(max_ind)

        # remove overlapped boxes
        overlapped_inds = np.where(ious[max_ind, :] > 0)[0]
        box_area[max_ind] = -1
        box_area[overlapped_inds] = -1

    return inds

# input: N x 4
# output: N x 2
def get_box_centers(boxes):

    bot_centers       =  boxes[:, :2].copy()
    bot_centers[:, 0] =  bot_centers[:, 0] + boxes[:, 2] / 2.
    bot_centers[:, 1] =  bot_centers[:, 1] + boxes[:, 3]

    return bot_centers

# load seq information from dataset
def load_seq_info(seq_info_file):
    seq_info = {}
    with open(seq_info_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            seps = line.strip().split('=')
            if seps[0] == 'imWidth': seq_info['w_img'] = int(seps[-1])
            elif seps[0] == 'imHeight': seq_info['h_img'] = int(seps[-1])
            elif seps[0] == 'seqLength': seq_info['seq_len'] = int(seps[-1])
            elif seps[0] == 'frameRate': seq_info['fps'] = int(seps[-1])
            elif seps[0] == 'name': seq_info['seq_name'] = seps[-1]
    return seq_info

import json
def save_to_json(file_name, dictionary, indent = 4):
    # Serializing json
    json_object = json.dumps(dictionary, indent=indent)
    # Writing to sample.json
    with open(file_name, "w+") as outfile:
        outfile.write(json_object)

def load_from_json(file_name):
    json_data = json.load(open(file_name))
    return json_data