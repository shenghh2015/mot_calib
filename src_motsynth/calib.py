import numpy as np
import cv2 as cv
from scipy import optimize

from tools.card_utils import CardBoard
from tools.camera_utils import Camera
from tools.geometry_utils import (gen_focal_length, Rotation, gen_surf_normal)

EPS = 1e-10

# init ECC module
number_of_iterations = 50
termination_eps = 0.01
# Define termination criteria
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

class MotionCalib:

    def __init__(self, cam_intr, cam_rot = np.array([0, 0, 0]), 
                                 cam_pos = np.array([0, 0, 0])):
        
        self.cam_intr = cam_intr
        self.cam_rot  = cam_rot
        self.cam_pos  = cam_pos
    
    # warp track boxes (tlwh) with ecc coefficients
    def warp_boxes(self, ecc_coef, track_boxes):
        ecc_coef    = np.expand_dims(ecc_coef, axis = 0)
        track_boxes = track_boxes.reshape((-1, 4))
        w_boxes = self.map_to_ref(ecc_coef, track_boxes)
        return w_boxes

    # compute ecc coefficients with a frame and the specified reference frame
    def compute_ecc(self, frame, ref, dsample = 4, 
                    mask = None, use_mask = False):
        h, w        = frame.shape[:2]
        h, w        = h // dsample, w // dsample
        frame       = cv.resize(frame, (w, h))
        ref         = cv.resize(ref,   (w, h))
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        gray_frame  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ref_frame   = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
        try:
            if not use_mask:
                (cc, warp_matrix) = cv.findTransformECC(gray_frame, ref_frame, warp_matrix, cv.MOTION_HOMOGRAPHY, criteria)
            else:
                mask = mask[::dsample, ::dsample]
                (cc, warp_matrix) = cv.findTransformECC(gray_frame, ref_frame, warp_matrix, cv.MOTION_HOMOGRAPHY, criteria, mask)
            
            self.warp_matrix = warp_matrix.copy()
            return warp_matrix
        except:

            print('ecc computation error')
            # return warp_matrix
            return self.warp_matrix.copy()

    def map_to_ref(self, ecc_coefs, tracks, dsample = 4):
        track_shape = tracks.shape
        if (ecc_coefs == np.eye(3, 3)).all(): return tracks

        if len(ecc_coefs) == 2: ecc_coefs = ecc_coefs.reshape((-1, 3, 3))

        if len(tracks.shape) == 2: tracks = tracks.reshape((-1, 1, 4))

        tracks    = tracks.copy() / dsample
        ecc_coefs = np.expand_dims(ecc_coefs, axis = 0)
        ecc_coefs = np.tile(ecc_coefs, (tracks.shape[0], 1, 1, 1))
        
        # track shape
        n, f   = tracks.shape[:2]

        mask   = tracks[:, :, -1] > 0

        tracks = tracks.reshape((-1, 4))

        t         = tracks[:, :2].copy()
        t[:, 0]  += tracks[:, 2] / 2
        
        r         = tracks[:, :2].copy()
        r[:, 0]  += tracks[:, 2]
        r[:, 1]  += tracks[:, 3] / 2

        b         = tracks[:, :2].copy()
        b[:, 0]  += tracks[:, 2] / 2
        b[:, 1]  += tracks[:, 3]

        l         = tracks[:, :2].copy()
        l[:, 1]  += tracks[:, 3] / 2

        trbl      = np.stack([t, r, b, l], axis = 0)
        trbl      = trbl.reshape((-1, 2))
        trbl      = np.concatenate([trbl, np.ones((4 * n * f, 1))], axis = -1)  # n * f * 4, 3
        trbl      = np.expand_dims(trbl, axis = -1)
        
        new_coefs = np.tile(ecc_coefs, (4, 1, 1)).reshape((-1, 3, 3))          # n * f * 4, 3, 3

        new_trbl  = np.matmul(new_coefs, trbl)                                 # n * f * 4, 3, 1
        
        new_trbl  = new_trbl.squeeze()
        new_trbl  = new_trbl[:, :2] / (new_trbl[:, 2:] + EPS)

        new_trbl  = new_trbl.reshape((4, n, f, 2))
        new_trbl  = new_trbl.transpose((1, 2, 0, 3))

        xmin, xmax  = new_trbl[:, :, :, 0].min(axis = -1), new_trbl[:, :, :, 0].max(axis = -1)
        ymin, ymax  = new_trbl[:, :, :, 1].min(axis = -1), new_trbl[:, :, :, 1].max(axis = -1)

        new_tracks  = np.zeros(new_trbl.shape[:3])
        ecc_tracks  = new_tracks.copy()
        new_tracks[:, :, 0] = xmin
        new_tracks[:, :, 1] = ymin
        new_tracks[:, :, 2] = xmax - xmin
        new_tracks[:, :, 3] = ymax - ymin

        ecc_tracks[mask]    = new_tracks[mask]

        ecc_tracks = ecc_tracks * dsample

        ecc_tracks = ecc_tracks.reshape(track_shape)

        return ecc_tracks

    # alpha, beta, x, y
    def mot_calib(self, det_boxes_list):

        # calibration
        cb_mapping_list = []
        for i in range(3):
            focal_length = gen_focal_length(w_img = 1080, self.cam_intr['fov'])
            camera       = Camera(fx = focal_length,           fy = focal_length,
                                  cx = self.cam_intr['w_img'], cy = self.cam_intr['h_img'])
            rotation   = Rotation(np.array(self.cam_rot))
            cb_mapping = CardBoard(self.cam_intr, rotation, self.cam_pos)
            cb_mapping_list.append(cb_mapping)
        cam_params = (self.cam_intr, self.cam_rot, self.cam_pos)
        params     = (det_boxes_list, cb_mapping_list, cam_params)
        initials   = [np.zeros((4,)), np.zeros((4,)), np.zeros((4,))]

        # initial loss
        initials   = [np.zeros((4,)), np.zeros((4,)), np.zeros((4,))]
        loss = calib_loss(initials, det_boxes_list, cb_mapping_list, cam_params)
        print(f'initial loss: {round(loss, 4)}')

        result     = optimize.minimize(calib_loss, initials, args = params, 
                        method = 'BFGS', options={'gtol': 1e-3, 'disp': False, 'maxiter': num_iters})
        print(result.x)

        # update
        self.cam_rot[:2] += result.x[-1][:2]
        self.cam_pos[:2] += result.x[-1][2:]
        print(f'update: {self.cam_rot}, {self.cam_pos}')

# variables: da, db, dx, dy
def calib_loss(variables, *args):

    boxes_list, cb_mapping_list, cam_params = args

    xyhw_list = []

    cam_intr, cam_rot, cam_pos = cam_params

    fov = cam_intr['fov']

    for boxes, cb_mapping, var in zip(boxes_list, cb_mapping_list, variables):

        ''' update aerial view '''
        # update rotation
        rotation  = cb_mapping.get_rotation()
        alpha     = cam_rot[0] + var[0]
        beta      = cam_rot[1] + var[1]
        rotation.set_angles(np.array([alpha, beta, 0]))

        pos_x       = cam_pos[0] + var[2]
        pos_y       = cam_pos[1] + var[3]
        new_cam_pos = np.array([pos_x, pos_y, cam_pos[-1]])

        # update camera and rotation in an aerial view object
        cb_mapping.set_camera(camera)
        cb_mapping.set_rotation(rotation)
        cb_mapping.set_pos(cam_pos[])

        ''' compute loss '''
        boxes  = boxes.reshape((-1, 4))
        xyhw   = cb_mapping.xyhw2bbox(xyhw)
        xyhw_list.append(xyhw)
    
    # height loss
    height_loss = np.abs((xyhw_list[0][:,2] - xyhw_list[1][:,2]) / xyhw_list[1][:,2]) + 
                  np.abs((xyhw_list[2][:,2] - xyhw_list[1][:,2]) / xyhw_list[1][:,2])

    # direction loss
    vel1     = xyhw_list[1][:,:2] - xyhw_list[0][:,:2]
    vel2     = xyhw_list[2][:,:2] - xyhw_list[1][:,:2]
    vel1     = vel1 / np.linalg.norm(vel1, axis = 1)
    vel2     = vel2 / np.linalg.norm(vel2, axis = 1)
    dir_loss = np.sum(vel1 * vel2, axis = 1)
    loss     = height_loss.mean() + dir_loss.mean()

    return loss