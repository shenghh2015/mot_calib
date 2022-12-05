# Author: Shenghua He (shenghh2015@gmail.com)
import numpy as np
from . import geometry_3d as gm3d

class Camera:
    
    def __init__(self, w_img = 1920, 
                       h_img = 1080, 
                       sx = 1152, 
                       sy = 1152,
                       cx = 960, 
                       cy = 540):

        self.w_img = w_img
        self.h_img = h_img

        self.sx = sx
        self.sy = sy
        self.cx = cx
        self.cy = cy
        # intrinsic matrix
        # self.Mint = np.array([[sx, 0, cx],
        #             [0, sy, cy],
        #             [0, 0,  1]])
        self.Mint = self._gen_intr_mat(np.array([sx, sy, cx, cy]))
        # inverse of intrinsic matrix
        # self.Mint_inv = np.array([[1./sx, 0, -cx/sx],
        #             [0, 1./sy, -cy/sy],
        #             [0, 0,  1]])
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

    # map 2d pts (N x 2) to 3d rays (3 x N)
    def map_2d_pts_to_3d_rays(self, pts_2d):
        U = np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis = -1)
        Xc = np.matmul(self.Mint_inv, U.T)
        return Xc

    # map 3d rays (3 x N) to 2d pts (N x 2)
    def map_3d_rays_to_2d_pts(self, rays_3d):
        U = np.matmul(self.Mint, rays_3d)
        V = U[:2, :] / U[2:, :]
        return V.T
    
    def update_intrisic(self, intr_vec):
        self.sx, self.sy, self.dx, self.dy = intr_vec.flatten()
        self.Mint = self._gen_intr_mat(intr_vec)
        self.Mint_inv = self._gen_intr_mat_inverse(intr_vec)


class ImageGroundFloor:

    def __init__(self, camera, rotation):
        self.camera = camera
        self.rotation = rotation

    # map 2d points in an image (img) plane onto the ground floor (grf)
    # input:  N x 2, return: N x 2
    def map_img_to_grf(self, img_pts_2d, axis = 2):
        Xc = self.camera.map_2d_pts_to_3d_rays(img_pts_2d)
        Xc = self.rotation.rotate(Xc)
        Xc = gm3d.normalize_rays_3d(Xc, axis = axis)
        inds = [i for i in range(Xc.shape[0]) if not i == axis]
        return Xc[inds, :].T

    # map 2d points on a ground floor into the image plane
    # input:  N x 2, return: N x 2
    def map_grf_to_img(self, grf_pts_2d):
        Xc = np.concatenate([grf_pts_2d, np.ones((grf_pts_2d.shape[0], 1))], axis = 1)
        Xc = self.rotation.rotate_inverse(Xc.T)
        U  = self.camera.map_3d_rays_to_2d_pts(Xc)
        return U
    
    def get_camera(self):
        return self.camera
    
    def get_rotation(self):
        return self.rotation
    
    def set_camera(self, camera):
        self.camera = camera
    
    def set_rotation(self, rotation):
        self.rotation = rotation