import numpy as np

EPS = 1e-8

class Rotation:

    def __init__(self, thetas = np.array([0, 0, 0])):

        self.thetas = deg_to_rad(thetas)
        self._gen_rot_mat()

    def _gen_rot_mat(self):

        R1 = np.array([
        [1, 0, 0],
        [0, np.cos(self.thetas[0]), -np.sin(self.thetas[0])],
        [0, np.sin(self.thetas[0]),  np.cos(self.thetas[0])]])

        R2 = np.array([
        [np.cos(self.thetas[1]), 0, np.sin(self.thetas[1])],
        [0, 1, 0],
        [-np.sin(self.thetas[1]), 0, np.cos(self.thetas[1])]])

        R3 = np.array([
        [np.cos(self.thetas[2]), -np.sin(self.thetas[2]), 0],
        [np.sin(self.thetas[2]), np.cos(self.thetas[2]), 0],
        [0, 0, 1]])

        R = np.matmul(R3, R2)
        R = np.matmul(R, R1)

        self.R = R
        self.R_inv = np.linalg.inv(R)
 
    def set_angles(self, thetas):
        self.thetas = deg_to_rad(thetas)
        # print(self.thetas)
        self._gen_rot_mat()

    def get_angles(self):
        return rad_to_deg(self.thetas.flatten())
    
    def get_R(self):
        return self.R
    
    def get_R_inv(self):
        return self.R_inv
    
    def set_R(self, rot_matrix):
        self.R     = rot_matrix.copy()
        self.R_inv = np.linalg.inv(self.R)

    def rotate(self, X):
        return np.matmul(self.R, X.T).T

    def rotate_inverse(self, X):
        return np.matmul(self.R_inv, X.T).T

# convert a radian vector to a degree vector
def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

# convert a degree vector to a radian vector
def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi

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