import numpy as np
from const import EPS

def gen_Mint(cam_info):
    return np.array([[cam_info['fx'], 0.,              cam_info['cx']],
                     [0.,              cam_info['fy'], cam_info['cy']],
                     [0.,              0.,             1.]],dtype = np.float32)

def gen_Mint_inv(cam_info):
    return np.array([[1./ cam_info['fx'], 0.,                 - cam_info['cx'] / cam_info['fx']],
                     [0.,                 1./ cam_info['fy'], - cam_info['cy'] / cam_info['fy']],
                     [0.,                 0.,                 1.]],dtype = np.float32)

def gen_Mint_inv_v2(focal_length, cx, cy):
    return np.array([[1./ focal_length, 0.,                   - cx / focal_length],
                     [0.,                 1./ focal_length,   - cy / focal_length],
                     [0.,                 0.,                 1.]],dtype = np.float32)

def gen_focal_length(w_img, fov):
    return w_img / np.tan(deg_to_rad(fov / 2)) / 2.

def compute_fov(w_img, focal_length):
    rad_fov = np.arctan(w_img / 2. / focal_length) * 2
    return rad_to_deg(rad_fov)

#   s: 0.28 - 1.86
# fov: 120 - 30
# s = 0.5 / tan(fov / 2)
def gen_focal_length_from_s(w_img, s):
    return w_img * s

def fov_to_s(fov):
    return 0.5 / np.tan(deg_to_rad(fov/ 2))

def s_to_fov(s):
    return 2 * rad_to_deg(np.arctan(0.5 / s))

# generate a roation matrix based on angle vectors
# thetas: angle vector 3 x 1
# return: rotation matrix 3 x 3 
def get_rot_R(thetas):

    # degree to rad (!!!)
    thetas =  deg_to_rad(thetas)

    R1 = np.array([
    [1, 0, 0],
    [0, np.cos(thetas[0]), -np.sin(thetas[0])],
    [0, np.sin(thetas[0]),  np.cos(thetas[0])]])

    R2 = np.array([
    [np.cos(thetas[1]), 0, np.sin(thetas[1])],
    [0, 1, 0],
    [-np.sin(thetas[1]), 0, np.cos(thetas[1])]])

    R3 = np.array([
    [np.cos(thetas[2]), -np.sin(thetas[2]), 0],
    [np.sin(thetas[2]), np.cos(thetas[2]), 0],
    [0, 0, 1]])

    R = np.matmul(R3, R2)
    R = np.matmul(R, R1)

    return R

def get_rot_R_inv(rot_matrix):
    return np.linalg.inv(rot_matrix)

# R: 3 x 3; rays_3d: N x 3
# output: N x 3 
def rotate(R, rays3d):
    rot_rays_3d = np.matmul(R, rays3d.T)
    return rot_rays_3d.T

# M_int: 3 x 3; pts2d: N x 2
# output: N x 3
def map_pts2d_to_rays3d(Mint_inv, pts2d):
    pts3d  = np.concatenate([pts2d, np.ones((pts2d.shape[0], 1))], axis = -1)
    rays3d = np.matmul(Mint_inv, pts3d.T)
    return rays3d.T

# M_int: 3 x 3; rays3d: N x 3
# output: N x 3
def map_rays3d_to_pts2d(Mint, rays3d):
    pts_3d = np.matmul(Mint, rays3d.T)
    pts_3d = homo2eclud(pts_3d.T)
    return pts_3d[:, :2]

# input: N x 3; output: N x 3s
def homo2eclud(rays3d):
    # a = rays3d.copy()
    # a[:, 0] = a[:, 0] / (EPS + a[:, -1])
    # a[:, 1] = a[:, 1] / (EPS + a[:, -1])
    # a[:, -1] = a[:, -1] / (EPS + a[:, -1])
    # return a
    new_rays3d = rays3d.copy()
    new_rays3d = new_rays3d / (new_rays3d[:, 2:3] + EPS)
    # print(pts3d)
    return new_rays3d[:, :2]

def eclud2homo(pts2d):
    pts3d = np.concatenate([pts2d, np.ones((pts2d.shape[0], 1))], axis = -1)
    return pts3d

def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi

# generate a surface normal based on alpha and beta
def gen_surf_normal(alpha, beta):
    alpha, beta = deg_to_rad(alpha), deg_to_rad(beta)
    a = - np.sin(beta)
    b = np.sin(alpha) * np.cos(beta)
    c = np.cos(alpha) * np.cos(beta)
    
    surf_normal = np.array([a, b, c]) if c > 0 else np.array([-a, -b, -c]) 

    return surf_normal

# rays: 3 x N x 3; surf_normal: 3 x 1
# heights N x 1
def get_vertical_heights(rays, surf_normal):

    # 3d rays related to top left, top right, bot center of bboxes
    top_left, top_right, bot_center   = rays[0, :], rays[1, :], rays[2, :]

    # surface normals
    surf_normal_bot = surf_normal.reshape((-1, 1))               # 3 x 1
    surf_normal_top = np.cross(top_left, top_right, axis = -1)   # N x 3 

    # compute heights
    M1      = np.expand_dims(np.sum(bot_center * surf_normal_top, axis = -1), 
                             axis = -1) # N x 1
    M2      = np.matmul(bot_center, surf_normal_bot)      # N x 1
    M3      = np.matmul(surf_normal_top, surf_normal_bot) # N x 1
    heights = M1 / M2 / M3                                # N x 1

    return heights

# compute heights with boxes and intrinsic matrix and surf_normal
def compute_heights(boxes, Mint_inv, surf_normal):

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

    rays3d     = map_pts2d_to_rays3d(Mint_inv, pts2d)
    rays3d     = rays3d.reshape((3, -1, 3))

    heights    = get_vertical_heights(rays3d, surf_normal)

    return heights

# map image points to BEV
def map_to_BEV(Mint_inv, rot_matrix, img_pts):

    img_pts    = img_pts.reshape((-1, 2))
    rays3d     = map_pts2d_to_rays3d(Mint_inv, img_pts)
    # print('new:')
    # print(rays3d[:10,:].T, rays3d.T.shape)
    rays3d_rot = rotate(rot_matrix, rays3d)
    # print(rays3d_rot[:10,:].T, rays3d_rot.T.shape)
    pts2d      = homo2eclud(rays3d_rot)

    return pts2d

# generate xyh related to bboxes according to cam_info and boxe tlwh
# cam_info: {'w_img': 1920, 
#            'h_img': 1080,  
#            'fov': 70, 
#            'alpha': 60, 
#            'beta': 0}
# boxes:     N x 4
def compute_xyh_from_boxes(cam_info, boxes):

    # camera parameters
    focal_length = gen_focal_length(w_img = cam_info['w_img'], 
                                    fov   = cam_info['fov'])
    # print(focal_length)
    cam_params   = {'fx': focal_length, 
                    'fy': focal_length, 
                    'cx': cam_info['w_img'] / 2, 
                    'cy': cam_info['h_img'] / 2}
    Mint_inv       = gen_Mint_inv(cam_params)

    surf_normal    = gen_surf_normal(cam_info['alpha'], cam_info['beta'])
    rot_matrix     = get_rot_R(np.array([cam_info['alpha'], cam_info['beta'], 0]))
    # rot_matrix_inv = get_rot_R_inv(rot_matrix)

    # print('new rotation: ', rot_matrix, rot_matrix.shape)

    # bottem center (x + w/2, y + h)
    bot_centers       =  boxes[:, :2].copy()
    bot_centers[:, 0] =  bot_centers[:, 0] + boxes[:, 2] / 2.
    bot_centers[:, 1] =  bot_centers[:, 1] + boxes[:, 3]

    # map to BEV 2d points
    xy = map_to_BEV(Mint_inv, rot_matrix, bot_centers)
    h  = compute_heights(boxes, Mint_inv, surf_normal).reshape((-1, 1))

    # print(h)
    # print(xy)
    return np.concatenate([xy, h], axis = -1)

def maps_xyh_to_img_pts(Mint, rot_matirx_inv, xyh):

    top_xyh      = xyh.copy()
    top_xyh[:, 2] = 1 - top_xyh[:, 2]

    bot_xyh = xyh.copy()
    bot_xyh[:, 2] = 1.

    xyhs    = np.concatenate([top_xyh, bot_xyh], axis = 0)

    rays    = rotate(rot_matirx_inv, xyhs)
    img_pts = map_rays3d_to_pts2d(Mint, rays)

    img_pts = img_pts.reshape((2, -1, 2))

    return img_pts.transpose((1, 0, 2))

def map_boxes_to_img_pts(cam_info, boxes):

    xyh     = compute_xyh_from_boxes(cam_info, boxes)

    focal_length = gen_focal_length(cam_info['w_img'], cam_info['fov'])
    
    cam_params   = {'fx': focal_length,
                    'fy': focal_length,
                    'cx': cam_info['w_img'] / 2,
                    'cy': cam_info['h_img'] / 2}
    Mint         = gen_Mint(cam_params)
    rot_matrix   = get_rot_R(np.array([cam_info['alpha'], cam_info['beta'], 0]))
    rot_matrix_inv = get_rot_R_inv(rot_matrix)
    img_pts = maps_xyh_to_img_pts(Mint, rot_matrix_inv, xyh)

    return img_pts
