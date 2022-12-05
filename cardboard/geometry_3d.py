import numpy as np

EPS = 1e-8

class Rotation:

    def __init__(self, thetas = np.array([0, 0, 0]), mode = 'XYZ'):
        self.thetas = deg_to_rad(thetas)
        self._gen_rot_mat(mode)

    def _gen_rot_mat(self, mode = 'XYZ'):

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

        # if mode == 'XYZ':
        #     R = np.matmul(R1, R2)
        #     R = np.matmul(R, R3)
        # else:
        R = np.matmul(R3, R2)
        R = np.matmul(R, R1)

        self.R = R
        self.R_inv = np.linalg.inv(R)
 
    def set_angles(self, thetas, mode = 'XYZ'):
        self.thetas = deg_to_rad(thetas)
        self._gen_rot_mat(mode)

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
        return np.matmul(self.R, X)

    def rotate_inverse(self, X):
        return np.matmul(self.R_inv, X)

# convert a radian vector to a degree vector
def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

# convert a degree vector to a radian vector
def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi

def gen_focal_length(w_img, fov):
    return w_img / np.tan(deg_to_rad(fov / 2)) / 2.
    
def gen_surf_normal(alpha, beta):
    alpha, beta = deg_to_rad(alpha), deg_to_rad(beta)
    a = - np.sin(beta)
    b = np.sin(alpha) * np.cos(beta)
    c = np.cos(alpha) * np.cos(beta)
    
    surf_normal = np.array([a, b, c]) if c > 0 else np.array([-a, -b, -c]) 

    return surf_normal

# normalize the rays with a specific axis for converting homogenous coordinates
# to ecludian coordinates
def normalize_rays_3d(X, axis = 1):
    a = X.copy()
    if axis == 2:
        a[0, :] = a[0, :] / (EPS + a[-1, :])
        a[1, :] = a[1, :] / (EPS + a[-1, :])
        a[-1,:] = a[-1,:] / (EPS + a[-1, :])
    elif axis == 1:
        a[0, :] = a[0, :] / (EPS + a[1, :])
        a[2, :] = a[2, :] / (EPS + a[1, :])
        a[1, :] = a[1, :] / (EPS + a[1, :])
    return a

# regress a plane from a set of 3d points
def regress_plane_from_3d_pts(pts_3d):
    pts_3d = pts_3d.reshape((3, -1))
    A = np.concatenate([pts_3d, np.ones((1, pts_3d.shape[-1]))])
    w, v = np.linalg.eig(np.matmul(A, A.T))
    min_index = np.argmin(w)
    return v[:, min_index]

# find rotation angles to generate a roation matrix that roates Z axis to a 
# target vector
def rotate_Z_to_trg_vec(surface_norm):
    surface_norm = surface_norm / np.linalg.norm(surface_norm)
    a, b, c = surface_norm.flatten()
    alpha  =  np.arcsin(-b)
    beta   =  np.arctan(a/ c)
    # adjust estimated beta
    if np.cos(alpha) * np.sin(beta) * a < 0 or\
       np.cos(alpha) * np.cos(beta) * c < 0: 
        beta = beta + np.pi if beta < 0 else beta - np.pi
    alpha = rad_to_deg(alpha)
    beta =  rad_to_deg(beta)
    return np.array([alpha, beta, 0])

# find rotation angles to generate a roation matrix that roates a target vector 
# to Z axis
def rotate_vec_to_Z(surface_norm):
    surface_norm = surface_norm / np.linalg.norm(surface_norm)
    a, b, c = surface_norm.flatten()
    beta    = np.arctan(-a)
    alpha   = np.arctan(b/c)

    # adjust estimated beta
    if np.sin(alpha) * np.cos(beta) * b < 0 or\
       np.cos(alpha) * np.cos(beta) * c < 0: 
        alpha = alpha + np.pi if alpha < 0 else alpha - np.pi
    alpha = rad_to_deg(alpha)
    beta =  rad_to_deg(beta)
    return np.array([alpha, beta, 0])

# compute the distances from 3d pts to a 3d plane
def measure_pts_to_plane_dis(pts_3d, plane_3d):
    plane_3d = plane_3d.reshape((1, 4))
    res = np.abs(np.matmul(plane_3d[:,:3], pts_3d) + plane_3d[0, 3])
    dis = res / np.linalg.norm(plane_3d[0, :3])
    return dis.T

''' target surface: a x + b y + c z + 1 = 0; surf_norm = [a, b, c]
the first ray determines the projected point on the target surface
the remaining two rays determine a surface that the trget vector insects with
return the heights of vectors starting from projected poinits to the intersections'''
# r1: the ray from box bottom center
# r2: the ray from box top left
# r3: the ray from box top right
# surf_norm: the surface norm of the target plane
def get_vertical_vec_heights(r1, r2, r3, surf_norm):

    n1 = surf_norm.reshape((-1, 1))    # surface norm of the target plane
    n2 = np.cross(r2, r3, axis = -1)   # N x 3, surface norms determined by r2, 
                                       # r3 and the optical center

    M1 = np.expand_dims(np.sum(r1 * n2, axis = -1), axis = -1) # N x 1
    M2 = np.matmul(r1, n1)             # N x 1
    M3 = np.matmul(n2, n1)             # N x 1
    hts = M1 / M2 / M3
    # hts = M1 * M2 / M3
    return hts

def get_vertical_vec_heights_v2(r1, r2, r3, surf_norm):

    n1 = surf_norm.reshape((-1, 1))    # surface norm of the target plane
    n2 = np.cross(r2, r3, axis = -1)   # N x 3, surface norms determined by r2, 
                                       # r3 and the optical center

    M1 = np.expand_dims(np.sum(r1 * n2, axis = -1), axis = -1) # N x 1
    M2 = np.matmul(r1, n1)             # N x 1
    M3 = np.matmul(n2, n1)             # N x 1
    hts = - M1 / M2 / M3
    # hts = M1 * M2 / M3
    return hts