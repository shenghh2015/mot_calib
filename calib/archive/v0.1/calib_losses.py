from calib.geometry import (compute_heights, 
                            gen_surf_normal,
                            gen_Mint_inv_v2,
                            gen_Mint,
                            get_rot_R,
                            get_rot_R_inv,
                            gen_focal_length,
                            maps_xyh_to_img_pts,
                            compute_xyh_from_boxes,
                            gen_focal_length_from_s, map_boxes_to_img_pts)
import numpy as np

# alpha and beta loss
def alpha_beta_loss(variables, *args):

    # variables
    alpha, beta = variables

    # parameters
    cam_params, boxes, s = args

    # inverse of intrinsic matrix
    focal_length = gen_focal_length_from_s(cam_params['w_img'], s)
    Mint_inv     = gen_Mint_inv_v2(focal_length, 
                               cam_params['w_img'] / 2,
                               cam_params['h_img'] / 2)

    # surface normal
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    # boxes
    boxes       = boxes.reshape((-1, 4))

    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()
    return h_loss

# s loss
def s_loss(variables, *args):

    # variables
    s = variables

    # parameters
    cam_params, boxes, alpha, beta = args
    # print(len(args))

    # inverse of intrinsic matrix
    focal_length = gen_focal_length_from_s(cam_params['w_img'], s)
    Mint_inv     = gen_Mint_inv_v2(focal_length, 
                                    cam_params['w_img'] / 2,
                                    cam_params['h_img'] / 2)

    # surface normal
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    # boxes
    boxes       = boxes.reshape((-1, 4))
    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()

    print(f's: {s} loss: {h_loss}')
    return h_loss

# loss related to joint optimization of alpha, beta and s
def alpha_beta_s_loss(variables, *args):

    # variables
    alpha, beta, s = variables

    # parameters
    cam_params, boxes = args

    # inverse of intrinsic matrix
    focal_length = gen_focal_length_from_s(cam_params['w_img'], s)
    Mint_inv     = gen_Mint_inv_v2(focal_length, 
                               cam_params['w_img'] / 2,
                               cam_params['h_img'] / 2)

    # surface normal
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    # boxes
    boxes       = boxes.reshape((-1, 4))

    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()

    return h_loss

# loss related to joint optimization of alpha, beta and s
def alpha_beta_fov_loss(variables, *args):

    # variables
    alpha, beta, fov = variables

    # parameters
    cam_params, boxes = args

    # inverse of intrinsic matrix
    focal_length = gen_focal_length(cam_params['w_img'], fov)
    Mint_inv     = gen_Mint_inv_v2(focal_length, 
                               cam_params['w_img'] / 2,
                               cam_params['h_img'] / 2)

    # surface normal
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    # boxes
    boxes       = boxes.reshape((-1, 4))

    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()

    return h_loss

def alpha_beta_fov_constrain_loss(variables, *args):

    height_loss    = alpha_beta_fov_loss(variables, *args)

    alpha, beta, fov  = variables
    cam_params,  boxes = args

    cam_info       = {'w_img': cam_params['w_img'],
                      'h_img': cam_params['h_img'],
                      'alpha': alpha,
                      'beta': beta,
                      'fov': fov}
    
    boxes  = boxes.reshape((-1, 4))
    c_loss = constrain_loss(cam_info, boxes)

    return height_loss + c_loss

def constrain_loss(cam_info, boxes):

    img_pts = map_boxes_to_img_pts(cam_info, boxes)
    top_pts = img_pts[:, 0, :]

    tl        = boxes[:, :2].copy()         # top left
    tr        = tl.copy()                   # top right
    tr[:, 0] += boxes[:, 2]

    # top left and top right distance
    left_se = (top_pts[:, 0] < tl[:, 0]) * (top_pts[:, 0] - tl[:, 0]) ** 2
    right_se = (top_pts[:, 0] > tr[:, 0]) * (top_pts[:, 0] - tr[:, 0]) ** 2

    loss    = np.sum((left_se + right_se)) / len(top_pts)

    return loss

# def constrained_alpha_beta_fov_loss(variables, *args):

#     alpha, beta, fov   = variables
#     cam_params,  boxes = args

#     cam_info    = {'w_img': cam_params['w_img'],
#                     'h_img': cam_params['h_img'],
#                     'alpha': alpha,
#                     'beta': beta,
#                     'fov': fov}

#     boxes        = boxes.reshape((-1, 2))
#     xyh          = compute_xyh_from_boxes(cam_info, boxes)

#     focal_length = gen_focal_length(cam_info['w_img'], cam_info['fov'])
#     cam_params   = {'fx': focal_length,
#                     'fy': focal_length,
#                     'cx': cam_info['w_img'] / 2,
#                     'cy': cam_info['h_img'] / 2}
#     Mint           = gen_Mint(cam_params)
#     rot_matrix     = get_rot_R(np.array([cam_info['alpha'], cam_info['beta'], 0]))
#     rot_matrix_inv = get_rot_R_inv(rot_matrix)
#     img_pts        = maps_xyh_to_img_pts(Mint, rot_matrix_inv, xyh)
