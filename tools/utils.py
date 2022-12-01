import numpy as np

def gen_focal_length(w_img, fov):
    return w_img / np.tan(deg_to_rad(fov / 2)) / 2.

def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi