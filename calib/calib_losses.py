from geometry import gen_focal_length
import numpy as np

def constrained_alpha_beta_fov_loss(variables, *args):

    ''' variables and parameters '''
    # variables
    alpha, beta, fov = variables

    # parameters
    a_view, boxes = args

    ''' update aerial view '''

    # update camera
    camera       = a_view.get_camera()
    focal_length = gen_focal_length(camera.w_img, fov)
    # print(camera.w_img, fov)
    cam_vec      = np.array([focal_length, focal_length, camera.w_img / 2, camera.h_img / 2])
    camera.update_intrisic(cam_vec)

    # update rotation
    rotation = a_view.get_rotation()
    rotation.set_angles(np.array([alpha, beta, 0]))

    # update camera and rotation in aerial view object
    a_view.set_camera(camera)
    a_view.set_rotation(rotation)

    ''' compute loss '''
    boxes   = boxes.reshape((-1, 4))
    xyh     = a_view.compute_xyh_from_boxes(boxes)    # 2N x 3
    # print(xyh.shape)
    img_pts = a_view.project_xyh_to_img(xyh)          #  N x 2 x 2

    xyh     = xyh.reshape((-1, 2, 3))                    #  N x 2 x 3
    
    ## height loss
    height_loss = np.abs((xyh[:, 0, 2] - xyh[:, 1, 2]) / xyh[:, 1, 2]).mean()

    ## contrain loss
    top_pts = img_pts[:, 0, :]

    tl        = boxes[:, :2].copy()         # top left
    tr        = tl.copy()                   # top right
    tr[:, 0] += boxes[:, 2]

    # top left and top right distance
    left_se = (top_pts[:, 0] < tl[:, 0]) * (top_pts[:, 0] - tl[:, 0]) ** 2
    right_se = (top_pts[:, 0] > tr[:, 0]) * (top_pts[:, 0] - tr[:, 0]) ** 2

    loss    = np.sum((left_se + right_se)) / len(top_pts)

    # print(loss)

    return loss + height_loss