from logging import logProcesses
from calib.calib_losses import alpha_beta_loss, alpha_beta_s_loss, s_loss, alpha_beta_fov_constrain_loss
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

def analyze_s_loss(alpha, beta, s_range, num_pts, *args):

    sol_space = np.linspace(start = s_range[0], stop = s_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for s in sol_space:
        loss = alpha_beta_s_loss([alpha, beta, s], cam_params, boxes)
        losses.append(loss)
    
    return losses, sol_space.tolist()

def analyze_s_loss_v2(alpha, beta, s_range, num_pts, *args):

    sol_space = np.linspace(start = s_range[0], stop = s_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for s in sol_space:
        loss = s_loss(s, cam_params, boxes, alpha, beta)
        losses.append(loss)
    
    return losses, sol_space.tolist()

def analyze_fov_loss(alpha, beta, fov_range, num_pts, *args):

    sol_space = np.linspace(start = fov_range[0], stop = fov_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for fov in sol_space:
        s = fov_to_s(fov)
        loss = s_loss(s, cam_params, boxes, alpha, beta)
        losses.append(loss)
    
    return losses, sol_space.tolist()

def analyze_fov_constrain_loss(alpha, beta, fov_range, num_pts, *args):

    sol_space = np.linspace(start = fov_range[0], stop = fov_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for fov in sol_space:
        loss = alpha_beta_fov_constrain_loss([alpha, beta, fov], cam_params, boxes)
        losses.append(loss)
    
    return losses, sol_space.tolist()

def analyze_alpha_loss(alpha_range, beta, s, num_pts, *args):

    sol_space = np.linspace(start = alpha_range[0], stop = alpha_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for sol in sol_space:
        loss = alpha_beta_s_loss([sol, beta, s], cam_params, boxes)
        losses.append(loss)
    
    return losses, sol_space.tolist()

def analyze_beta_loss(alpha, beta_range, s, num_pts, *args):

    sol_space = np.linspace(start = beta_range[0], stop = beta_range[1], num = num_pts)
    
    losses = []

    cam_params, boxes = args[0]

    for sol in sol_space:
        loss = alpha_beta_s_loss([alpha, sol, s], cam_params, boxes)
        losses.append(loss)
    
    return losses, sol_space.tolist()


def analyze_loss(bounds, num_pts, *args):

    alpha_bounds, beta_bounds, s_bounds = bounds

    alpha_space = np.linspace(start = alpha_bounds[0], stop = alpha_bounds[1], num = num_pts[0])
    beta_space  = np.linspace(start = beta_bounds[0],  stop = beta_bounds[1], num = num_pts[1])
    s_space     = np.linspace(start = s_bounds[0],     stop = s_bounds[1], num = num_pts[2])
    
    losses = np.zeros(num_pts)

    cam_params, boxes = args[0]

    for ia, alpha in enumerate(alpha_space):

        for ib, beta in enumerate(beta_space):

            for ind, s in enumerate(s_space):

                alpha, beta, s = alpha_space[ia], beta_space[ib], s_space[ind]
                loss = alpha_beta_s_loss([alpha, beta, s], cam_params, boxes)
                # losses.append(loss)
                losses[ia, ib, ind] = loss
    
    min_inds = np.unravel_index(np.argmin(losses, axis=None), losses.shape)
    print(min_inds)
    a_ind, b_ind, s_ind = min_inds
    res      = (alpha_space[a_ind], beta_space[b_ind], s_space[s_ind])
    
    return res, losses[a_ind, b_ind, s_ind]

def get_box_pts(boxes, 
                loc = 'bottom'):
    if loc == 'center':
        box_pts = boxes[:, :2] + boxes[:, 2:] / 2
    elif loc == 'bottom':
        box_pts = boxes[:, :2].copy()
        box_pts[:, 0] += boxes[:, 2] / 2 # x + w/2
        box_pts[:, 1] += boxes[:, 3]     # y + h
    return box_pts

def fov_constrain_loss(aview, rotation, boxes):

    bot_centers  = get_box_pts(boxes)

    xy           = aview.map_img_to_grf(bot_centers, axis = 2)
    h            = aview.compute_heights(boxes).reshape((-1, 1))
    xyh          = np.concatenate([xy, h], axis = -1)

    top_xyh      = xyh.copy()
    top_xyh[:, 2] = 1 - top_xyh[:, 2]

    bot_xyh = xyh.copy()
    bot_xyh[:, 2] = 1.

    xyhs    = np.concatenate([top_xyh, bot_xyh], axis = 0)
    rays    = rotation.rotate_inverse(xyhs.T)
    img_pts = aview.camera.map_3d_rays_to_2d_pts(rays)

    top_pts = img_pts[:len(top_xyh), :]

    tl        = boxes[:, :2].copy()
    tr        = tl.copy()
    tr[:, 0] += boxes[:, 2]

    left_se = (top_pts[:, 0] < tl[:, 0]) * (top_pts[:, 0] - tl[:, 0]) ** 2
    right_se = (top_pts[:, 0] > tr[:, 0]) * (top_pts[:, 0] - tr[:, 0]) ** 2

    loss    = np.sum((left_se + right_se)) / len(top_pts)

    return loss
