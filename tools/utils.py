import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

def gen_focal_length(w_img, fov):
    return w_img / np.tan(deg_to_rad(fov / 2)) / 2.

def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi

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