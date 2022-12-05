from calib.geometry import (compute_heights, 
                            gen_surf_normal, 
                            gen_Mint_inv, 
                            gen_Mint_inv_v2,
                            gen_focal_length,
                            gen_focal_length_from_s)
import numpy as np

# alpha and beta loss
def alpha_beta_loss(variables, *args):

    # variables
    alpha, beta = variables

    # parameters
    cam_params, boxes = args

    # camera parameters
    focal_length = gen_focal_length(cam_params['w_img'], cam_params['fov'])
    cam_info     = {'fx': focal_length, 
                    'fy': focal_length,
                    'cx': cam_params['w_img'] / 2,
                    'cy': cam_params['h_img'] / 2}

    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv(cam_info)
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal
    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 4))
    h_dif       = np.abs((heights[:, 1] - heights[:, 3]) / heights[:, 3])
    h_loss      = h_dif.mean()
    return h_loss

# fov loss
def fov_loss(variables, *args):

    # variables
    fov = variables

    # parameters
    cam_params, boxes, alpha, beta = args

    # camera parameters
    focal_length = gen_focal_length(cam_params['w_img'], fov)

    cam_info    = {'fx': focal_length, 
                   'fy': focal_length,
                   'cx': cam_params['w_img'] / 2,
                   'cy': cam_params['h_img'] / 2}
    
    print(alpha, beta, fov)
    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv(cam_info)
    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal
    heights     = compute_heights(boxes, Mint_inv, surf_normal)
    heights     = heights.reshape((-1, 4))
    h_dif       = np.abs((heights[:, 1] - heights[:, 3]) / heights[:, 3])
    h_loss      = h_dif.mean()
    return h_loss

def alpha_beta_fov_loss(variables, *args):

    # variables
    alpha, beta, fov = variables

    # parameters
    cam_params, boxes = args

    # camera parameters
    focal_length = gen_focal_length(cam_params['w_img'], fov)

    cam_info    = {'fx': focal_length, 
                   'fy': focal_length,
                   'cx': cam_params['w_img'] / 2,
                   'cy': cam_params['h_img'] / 2}
    
    print(alpha, beta, fov)
    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv(cam_info)

    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    heights     = compute_heights(boxes, Mint_inv, surf_normal)

    heights     = heights.reshape((-1, 4))
    h_dif       = np.abs((heights[:, 1] - heights[:, 3]) / heights[:, 3])
    h_loss      = h_dif.mean()
    return h_loss

def alpha_beta_fov_loss_v2(variables, *args):

    # variables
    alpha, beta, fov = variables

    # parameters
    cam_params, boxes = args

    # camera parameters
    focal_length = gen_focal_length(cam_params['w_img'], fov)

    # cam_info    = {'fx': focal_length, 
    #                'fy': focal_length,
    #                'cx': cam_params['w_img'] / 2,
    #                'cy': cam_params['h_img'] / 2}
    
    # print(alpha, beta, fov)
    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv_v2(focal_length,
                                  cam_params['w_img'] / 2, cam_params['h_img'] / 2)

    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    heights     = compute_heights(boxes, Mint_inv, surf_normal)

    heights     = heights.reshape((-1, 4))
    h_dif       = np.abs((heights[:, 1] - heights[:, 3]) / heights[:, 3])
    h_loss      = h_dif.mean()

    return h_loss


def alpha_beta_fov_loss_v3(variables, *args):

    # variables
    alpha, beta, fov = variables

    # parameters
    cam_params, boxes = args

    # camera parameters
    focal_length = gen_focal_length(cam_params['w_img'], fov)

    # cam_info    = {'fx': focal_length, 
    #                'fy': focal_length,
    #                'cx': cam_params['w_img'] / 2,
    #                'cy': cam_params['h_img'] / 2}
    
    # print(alpha, beta, fov)
    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv_v2(focal_length,
                                  cam_params['w_img'] / 2, cam_params['h_img'] / 2)

    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    heights     = compute_heights(boxes, Mint_inv, surf_normal)

    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()

    return h_loss


def alpha_beta_fov_loss_v4(variables, *args):

    # variables
    alpha, beta, s = variables

    # parameters
    cam_params, boxes = args

    # camera parameters
    focal_length = gen_focal_length_from_s(cam_params['w_img'], s)
    
    print(alpha, beta, s)
    boxes       = boxes.reshape((-1, 4))
    Mint_inv    = gen_Mint_inv_v2(focal_length,
                                  cam_params['w_img'] / 2, cam_params['h_img'] / 2)

    surf_normal = gen_surf_normal(alpha, beta)
    if surf_normal[2] < 0: surf_normal = - surf_normal

    heights     = compute_heights(boxes, Mint_inv, surf_normal)

    heights     = heights.reshape((-1, 2))
    h_dif       = np.abs((heights[:, 0] - heights[:, 1]) / heights[:, 1])
    h_loss      = h_dif.mean()

    return h_loss