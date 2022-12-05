import numpy as np
import cv2 as cv
from calib.geometry import gen_focal_length 
from scipy import optimize

from cardboard.new_cardboard_v4 import CardBoard
from cardboard.camera_utils import Camera
from cardboard.geometry_3d import Rotation, gen_surf_normal
from calib.kalman_filter import KalmanFilter
EPS = 1e-10

# init ECC module
number_of_iterations = 50
termination_eps = 0.01
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

class IncreCalib():

    def __init__(self, alpha = 0, beta = 0, fov = 0):
        self.alpha = alpha
        self.beta  = beta
        self.fov   = fov
        self.x     = 0
        self.y     = 0
        self.cam_height = 1.

        #TO-DO: set the covariance
        self.kalman = KalmanFilter()
        self.mean, self.cov = self.kalman.initiate(np.array([self.alpha, 
                                                   self.beta, self.x, self.y]))
        
        self.pred   = self.mean[:4].copy()
        self.meas   = self.mean[:4].copy()
        self.update = self.mean[:4].copy()

        self.warp_matrix = np.eye(3, 3, dtype=np.float32)

        # history
        self.pred_list   = []
        self.meas_list   = []
        self.update_list = []
    
    # warp track boxes (tlwh) with ecc coefficients
    def warp_boxes(self, ecc_coef, track_boxes):
        ecc_coef    = np.expand_dims(ecc_coef, axis = 0)
        track_boxes = track_boxes.reshape((-1, 4))
        w_boxes = self.map_to_ref(ecc_coef, track_boxes)
        return w_boxes

    # compute ecc coefficients with a frame and the specified reference frame
    def compute_ecc(self, frame, ref, dsample = 4, mask = None, use_mask = False):
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
    
    def update_cam_states(self, cb_mapping):

        # prediction
        self.mean, self.cov = self.kalman.predict(self.mean, self.cov)
        self.pred = self.mean.copy()
        
        # measure
        self.meas = np.array([self.alpha, self.beta, self.x, self.y])

        # update
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, self.meas)
        self.update = self.mean.copy()
        self.alpha, self.beta, self.x, self.y = self.update

        # camera       = cb_mapping.get_camera()
        # focal_length = gen_focal_length(camera.w_img, self.fov)
        # cam_vec      = np.array([focal_length, focal_length, camera.w_img / 2, camera.h_img / 2])
        # camera.update_intrisic(cam_vec)

        # update rotation
        rotation     = cb_mapping.get_rotation()
        rotation.set_angles(np.array([self.alpha, self.beta, 0]), mode = 'ZYX')

        # update camera and rotation in an aerial view object
        cb_mapping.set_rotation(rotation)
        cb_mapping.cam_height = self.cam_height

        # update history
        self.pred_list.append(self.pred.copy())
        self.meas_list.append(self.meas.copy())
        self.update_list.append(self.update.copy())

        return cb_mapping

    def update_calib_GD(self, boxes, track_boxes, cb_mapping, num_iters = 5):

        cur_cam_params = (self.fov,)
        xyhw           = cb_mapping.bbox2xyhw(track_boxes)          # 3d projections of track boxes

        # test the initial loss
        # initials = [0., 0., 0., 0.,]
        initials      = self.update.copy()
        loss          = ecc_alpha_beta_fov_dt_loss(initials, boxes, xyhw, cb_mapping, cur_cam_params)
        print(f'initial loss xx: {round(loss, 4)}')

        # create an empty mapping object
        new_camera    = Camera()
        new_rotation  = Rotation()
        new_mapping   = CardBoard(new_camera, new_rotation)
        params        = (boxes, xyhw, new_mapping, cur_cam_params)

        print('begin the incremental')
        result         = optimize.minimize(ecc_alpha_beta_fov_dt_loss, initials, args = params, 
                        method = 'BFGS', options={'gtol': 1e-3, 'disp': False, 'maxiter': num_iters})

        d_alpha, d_beta, dx, dy = result.x
        d_fov = 0
        dz = 0

        alpha, beta, fov             = self.alpha, self.beta, self.fov
        new_alpha, new_beta, new_fov = alpha + d_alpha, beta + d_beta, fov + d_fov

        print(f'd_alpha {round(d_alpha, 4)} d_beta {round(d_beta, 4)} d_fov {round(d_fov, 4)}')
        print(f'dx {round(dx, 4)} dy {round(dy, 4)} dz {round(dz, 4)} loss {result.fun}')
        print(f'updated alpha {round(new_alpha, 4)} beta {round(new_beta, 4)} fov {round(new_fov, 4)}, \
                T = {[round(dx, 4), round(dy, 4), round(dz, 4)]}')

        # d_alpha, d_beta, d_fov, dx, dy, dz = 0, 0, 0, 0, 0, 0
        alpha, beta, fov             = self.alpha, self.beta, self.fov
        new_alpha, new_beta, new_fov = alpha + d_alpha, beta + d_beta, fov + d_fov

        # print(new_alpha, new_beta, new_fov, d_alpha, d_beta, d_fov, dx, dy, dz)
        # generate a cardboard mapping based on optimized parameters
        camera       = new_mapping.get_camera()
        focal_length = gen_focal_length(camera.w_img, new_fov)
        cam_vec      = np.array([focal_length, focal_length, camera.w_img / 2, camera.h_img / 2])
        camera.update_intrisic(cam_vec)

        # update rotation
        rotation   = new_mapping.get_rotation()
        rotation.set_angles(np.array([new_alpha, new_beta, 0]), mode = 'ZYX')

        # update camera and rotation in an aerial view object
        new_mapping.set_camera(camera)
        new_mapping.set_rotation(rotation)
        new_mapping.set_transition(np.array([dx, dy, dz]))

        # compute camera height
        cam_height     = cb_mapping.cam_height
        # d_rotation     = Rotation(np.array([d_alpha, d_beta, 0]), mode = 'ZYX')
        # d_T            = np.array([dx, dy, dz])
        # surf_n         = gen_surf_normal(alpha, beta)
        # new_cam_height = abs(cam_height - (surf_n * d_T).sum())
        new_cam_height = cam_height - dz

        self.alpha      = new_alpha
        self.beta       = new_beta
        self.fov        = new_fov
        self.cam_height = new_cam_height

        return new_mapping, np.array([dx, dy, dz]), result.fun
    
    def get_calib_results(self):
        return np.array([self.alpha, self.beta, self.fov])

def tlwh2rect2d(boxes):
    
    rect2d = np.zeros(boxes.shape + (2,), dtype = np.float32)

    tl        = boxes[:, :2].copy()

    tr        = tl.copy()
    tr[:, 0] += boxes[:, 2]

    br        = tl.copy()
    br[:, :]  += boxes[:, 2:]

    bl        = tl.copy()
    bl[:, 1] += boxes[:, 3]

    rect2d[:, 0, :] = tl
    rect2d[:, 1, :] = tr
    rect2d[:, 2, :] = br
    rect2d[:, 3, :] = bl

    return rect2d

# corner loss
# variables: alpha, beta, x, y
def ecc_alpha_beta_fov_dt_loss(variables, *args):

    alpha, beta, x, y = variables
    
    d_fov = 0
    dz = 0

    # parameters
    boxes, xyhw, cb_mapping, initials = args

    # udpate camera and rotation
    fov = initials

    ''' update aerial view '''
    # update camera
    camera       = cb_mapping.get_camera()
    focal_length = gen_focal_length(camera.w_img, fov)
    cam_vec      = np.array([focal_length, focal_length, camera.w_img / 2, camera.h_img / 2])
    camera.update_intrisic(cam_vec)

    # update rotation
    rotation     = cb_mapping.get_rotation()
    rotation.set_angles(np.array([alpha, beta, 0]), mode = 'ZYX')

    # update camera and rotation in an aerial view object
    cb_mapping.set_camera(camera)
    cb_mapping.set_rotation(rotation)
    cb_mapping.set_transition(np.array([x, y, 0]))

    ''' compute loss '''
    boxes           = boxes.reshape((-1, 4))
    new_boxes       = cb_mapping.xyhw2bbox(xyhw)

    # box_centers     = boxes[:,     :2] + boxes[:, 2:] / 2
    # new_box_centers = new_boxes[:, :2] + new_boxes[:, 2:] / 2

    # tlbr
    box_corners = boxes.copy()
    new_box_corners = new_boxes.copy()
    box_corners[:, 2:] += box_corners[:, :2]
    new_box_corners[:, 2:] += new_box_corners[:, :2] 

    box_corners = boxes.reshape((-1, 2))
    new_box_corners = new_boxes.reshape((-1, 2))

    loss            = np.linalg.norm(box_corners - new_box_corners, axis = 1).mean()

    return loss