# Author: Shenghua He (shenghh2015@gmail.com)
import numpy as np
from geometry import (get_vertical_heights, 
                      gen_surf_normal)

from const import EPS

class Camera:
    
    def __init__(self, w_img = 1920, h_img = 1080, sx = 1152, sy = 1152, 
                       cx = 960, cy = 540):

        # camera parameters
        self.w_img = w_img
        self.h_img = h_img
        self.sx = sx
        self.sy = sy
        self.cx = cx
        self.cy = cy

        # intrinsic matrix
        self.Mint = self._gen_intr_mat(np.array([sx, sy, cx, cy]))

        # inverse of intrinsic matrix
        self.Mint_inv = self._gen_intr_mat_inverse(np.array([sx, sy, cx, cy]))
    
    def _gen_intr_mat(self, intr_vec):
        sx, sy, cx, cy = intr_vec.flatten()
        Mint = np.array([[sx, 0, cx],
                        [0, sy, cy],
                        [0, 0,  1]])
        return Mint
    
    def _gen_intr_mat_inverse(self, intr_vec):
        sx, sy, cx, cy = intr_vec.flatten()
        Mint_inv = np.array([[1./sx, 0, -cx/sx],
                            [0, 1./sy, -cy/sy],
                            [0, 0,  1]])
        return Mint_inv

    # map 2d pts (N x 2) to 3d rays (Nx3)
    def map_pts2d_to_rays3d(self, pts_2d):
        pts_3d = np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis = -1)
        return np.matmul(self.Mint_inv, pts_3d.T).T

    # map 3d rays (N x 3) to 2d pts (N x 2)
    def map_rays3d_to_pts2d(self, rays_3d):
        pts_3d = np.matmul(self.Mint, rays_3d.T).T
        pts_3d = pts_3d / (pts_3d[:, 2:] + EPS)
        
        return pts_3d[:, :2]
    
    # update intrinsic matrix and its inverse
    def update_intrisic(self, intr_vec):
        self.sx, self.sy, self.cx, self.cy = intr_vec.flatten()
        self.Mint = self._gen_intr_mat(intr_vec)
        self.Mint_inv = self._gen_intr_mat_inverse(intr_vec)



class AerialView:

    def __init__(self, camera, rotation):
        self.camera = camera
        self.rotation = rotation

        # compute surface normal
        angles = rotation.get_angles()
        self.surf_normal = gen_surf_normal(angles[0], angles[1])

    # map 2d points in an image (img) plane onto the ground floor (grf)
    # input:  N x 2, return: N x 2
    def map_img_to_bev(self, img_pts):
        rays  = self.camera.map_pts2d_to_rays3d(img_pts)
        rays  = self.rotation.rotate(rays)
        pts3d = rays / (rays[:, 2:] + EPS)
        return pts3d[:, :2]

    # map 2d points on a ground floor into the image plane
    # input:  N x 2, return: N x 2
    def map_bev_to_img(self, pts_2d):
        rays = np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis = 1)
        rays = self.rotation.rotate_inverse(rays)
        return self.camera.map_rays3d_to_pts2d(rays)

    # compute heights of persons in reference to a 3d plane determined by 
    # a x + b y + c z = 1 (c > 0)
    # compute heights with boxes and intrinsic matrix and surf_normal
    def compute_heights_from_boxes(self, boxes):
        
        # boxes
        boxes  = boxes.reshape((-1, 4))

        # bottem center (x + w/2, y + h)
        bot_centers       =  boxes[:, :2].copy()
        bot_centers[:, 0] =  bot_centers[:, 0] + boxes[:, 2] / 2.
        bot_centers[:, 1] =  bot_centers[:, 1] + boxes[:, 3]

        # box top left  (x, y)
        top_left          =  boxes[:, :2].copy()            

        # box top right (x + w, y)
        top_right       =  boxes[:, :2].copy()
        top_right[:, 0] =  top_right[:, 0] + boxes[:, 2]

        pts2d      = np.concatenate([top_left, top_right, bot_centers])

        rays3d     = self.camera.map_pts2d_to_rays3d(pts2d)
        rays3d     = rays3d.reshape((3, -1, 3))

        # print(self.surf_normal)
        heights    = get_vertical_heights(rays3d, self.surf_normal)

        return heights
    
    # compute xyh
    # boxes: N x 4
    def compute_xyh_from_boxes(self, boxes):

        bot_centers        = boxes[:, :2].copy()
        bot_centers[:, 0] += boxes[:, 2] / 2
        bot_centers[:, 1] += boxes[:, 3]

        xy  = self.map_img_to_bev(bot_centers)
        
        h   = self.compute_heights_from_boxes(boxes).reshape((-1, 1))

        xyh = np.concatenate([xy, h], axis = -1)

        return xyh
    
    # project xyh to image plane
    def project_xyh_to_img(self, xyh):

        # top xyz
        top_xyh = xyh.copy()
        top_xyh[:, 2] = 1 - top_xyh[:, 2]

        # bot xyz
        bot_xyh = xyh.copy()
        bot_xyh[:, 2] = 1.

        xyhs    = np.concatenate([top_xyh, bot_xyh], axis = 0)
        # print(self.rotation.get_angles())
        # print(self.rotation.get_R_inv())
        # print(xyhs)
        rays    = self.rotation.rotate_inverse(xyhs)
        # print(rays)
        img_pts = self.camera.map_rays3d_to_pts2d(rays)

        img_pts = img_pts.reshape((2, -1, 2))
        return img_pts.transpose((1, 0, 2))

    def get_camera(self):
        return self.camera
    
    def get_rotation(self):
        return self.rotation
    
    def set_camera(self, camera):
        self.camera = camera
    
    def set_rotation(self, rotation):
        self.rotation = rotation
        alpha, beta, _= rotation.get_angles()
        self.surf_normal = gen_surf_normal(alpha, beta)