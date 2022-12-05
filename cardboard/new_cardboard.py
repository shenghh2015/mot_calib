import yolox.camera.geometry_3d as gm3d
import numpy as np

class CardBoard:

    def __init__(self, camera, rotation, cam_height = 1.):

        self.camera      = camera
        self.rotation    = rotation
        self.cam_height  = cam_height
    
    # reject the bboxes that are above vanishing line
    # return the indices of boxes that are below the vanishing line
    def reject_bboxes_above_horizon(self, bboxes):

        # botteom cetners of bboxes in image plane
        bcs = bboxes[:, :2].copy()
        bcs[:, 0] += bboxes[:, 2] / 2
        bcs[:, 1] += bboxes[:, 3]

        # 3d rays in camera coordinate system
        bc_rays = self.camera.map_2d_pts_to_3d_rays(bcs)
        
        # 3d rays in aerial view
        bc_rays = self.trans_to_aerial(bc_rays)

        return np.where(bc_rays[:, -1] > 0)[0]
    
    # reject the objects that are larger than some sizes
    def reject_bboxes_by_sizes(self, bboxes, h_mean, h_std, num_std):

        # aspect ratio
        a_mask = bboxes[:, -2] < bboxes[:, -1]
        
        cboards = self.bboxes2cboards(bboxes)
        zyzhw = cboard2xyzhw(cboards)

        h_mask = np.logical_and(zyzhw[:, -2] < h_mean + num_std * h_std,
                                      zyzhw[:, -2] > h_mean - num_std * h_std)

        return np.where(np.logical_and(a_mask, h_mask))[0]

    # reject the objects that are larger than some sizes
    def reject_bboxes_by_sizes_v2(self, bboxes, h_lower = 0.25, h_upper = 1.0):

        # aspect ratio
        a_mask = bboxes[:, -2] < bboxes[:, -1]
        
        cboards = self.bboxes2cboards(bboxes)
        zyzhw = cboard2xyzhw(cboards)

        h_mask = np.logical_and(zyzhw[:, -2] <= h_upper,
                                      zyzhw[:, -2] >= h_lower)

        return np.where(np.logical_and(a_mask, h_mask))[0]

    def reject_bboxes_by_bottom(slef, bboxes, h_img):
        return np.where((bboxes[:, 1] + bboxes[:, 3]) <= h_img)[0]

    def bboxes2quadrangles3d(self, bboxes):

        bboxes = bboxes.reshape((-1, 4))
        # top left, top right, bottom right, bottom left, bottom centers: N x 2
        tls  = bboxes[:, :2]                  
        brs  = bboxes[:, :2] + bboxes[:, 2:]

        trs  = tls.copy()
        trs[:, 0] += bboxes[:, 2]

        bls  = tls.copy()
        bls[:, 1] += bboxes[:, 3]

        # 3d rays related to top left, top right, bottom right, bottom left, bottom center of boxes
        pts_2d      = np.concatenate([tls, trs, brs, bls], axis = 0)       # 4 x N x 3
        rays        = self.camera.map_2d_pts_to_3d_rays(pts_2d)            # 4 x N x 3
        rot_rays    = self.rotation.rotate(rays)
        grf_corners = gm3d.normalize_rays_3d(rot_rays, axis = 2)
        grf_corners = grf_corners.T
        grf_corners = grf_corners.reshape((4, -1, 3))

        line_vect   = grf_corners[2,:,:] - grf_corners[3,:,:]
        line_vect   = line_vect / np.linalg.norm(line_vect, axis = 1, keepdims = True)
        surf_n2     = np.cross(line_vect, np.array([0, 0, 1]))
        surf_n2     = surf_n2 / np.linalg.norm(surf_n2, axis = 1, keepdims = True)
        pos_x_msk   = surf_n2[:, 0:1] > 0
        surf_n2     = surf_n2 * pos_x_msk + surf_n2 * (pos_x_msk - 1.)

        ds          = (grf_corners[2, :] * surf_n2).sum(axis = 1, keepdims = True)

        grf_corners = grf_corners.reshape((-1, 3))
        scales      = np.tile(ds, (4, 1)) / (grf_corners * np.tile(surf_n2, (4, 1))).sum(axis = 1, keepdims = True)
        pts_3d      = scales * grf_corners

        # generate rectangle cardboards
        quadrangle_cboards = pts_3d.reshape(4, -1, 3)
        quadrangle_cboards = quadrangle_cboards.transpose([1, 0, 2])

        return quadrangle_cboards, line_vect, surf_n2

    # map boxes in image plane to cardboard in aerial view
    # inputs:  bboxes N x 4 (tlwh)
    # outputs: cardboards N x 4 x 3
    def bboxes2cboards(self, bboxes):
        bboxes = bboxes.reshape((-1, 4))
        quad3d, base_u, base_v  = self.bboxes2quadrangles3d(bboxes)
        # print(quad3d.shape, base_u.shape, base_v.shape)
        cboards = cut_quad3d(quad3d, base_u, base_v)
        return cboards
    
    def cboards2quadrangles2d(self, cardboards):
        # translate the cardboard coordinates back to the camera coordinate system
        pts_3d       = cardboards.reshape((-1, 3))
        pts_3d       = self.rotation.rotate_inverse(pts_3d.T)
        pts_2d       = self.camera.map_3d_rays_to_2d_pts(pts_3d)
        # generate rectangle bounding boxes
        quadrangle_bboxes = pts_2d.reshape((-1, 4, 2))
        return quadrangle_bboxes

    # map cardboard in the aerial view to the bboxes in the image plane
    # input: cardboards N x 4 x 3 
    # output: boxes N x 4 (tlwh)
    def cboards2bboxes(self, cboards):
        cboards = cboards.reshape((-1, 4, 3))
        quad2d  = self.cboards2quadrangles2d(cboards)
        return get_max_2d_tlwhs(quad2d)
    
    # map from image to ground floor
    def map_img_pts_to_grf_pts(self, box_pts):
        box_pts = box_pts.reshape((-1, 2))
        ray_3d  = self.camera.map_2d_pts_to_3d_rays(box_pts)
        ray_3d  = self.rotation.rotate(ray_3d)
        xyz     = gm3d.normalize_rays_3d(ray_3d, axis = 2).T
        return xyz[:, :2]
    
    def map_grf_pts_to_img_pts(self, xy):
        xy      = xy.reshape((-1, 2))
        pts3d   = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis = -1)
        pts3d   = self.rotation.rotate_inverse(pts3d.T)
        img_pts = self.camera.map_3d_rays_to_2d_pts(pts3d)
        return img_pts

    def xyhw2cboard(self, xyhw):
        
        xy         = xyhw[:, :2]
        img_pts    = self.map_grf_pts_to_img_pts(xy)

        line_left       = img_pts.copy()
        line_left[:, 0] = 0
        line_right      = img_pts.copy() 
        line_right[:, 0] = self.camera.w_img
        
        lines_2d   = np.concatenate([line_left, line_right], axis = 0)
        lines_grf  = self.map_img_pts_to_grf_pts(lines_2d)
        lines_grf  = lines_grf.reshape((2, -1, 2))
        lines_vecs = lines_grf[0, :, :] - lines_grf[1, :, :]

        lines_vecs_3d = np.concatenate([lines_vecs, np.zeros((lines_vecs.shape[0], 1))], axis = -1)
        lines_vecs_3d = lines_vecs_3d / np.linalg.norm(lines_vecs_3d, axis = 1, keepdims = True)

        pos_x_msk     = lines_vecs_3d[:, 0:1] > 0
        lines_vecs_3d = lines_vecs_3d * pos_x_msk + lines_vecs_3d * (pos_x_msk - 1.)
        
        xyz    = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis = -1)

        w   = xyhw[:, 3:]
        bls = xyz - lines_vecs_3d * w / 2
        brs = xyz + lines_vecs_3d * w / 2

        tls = bls.copy()
        tls[:, 2] -= xyhw[:, 2]

        trs = brs.copy()
        trs[:, 2] -= xyhw[:, 2]
        
        cboards = np.stack([tls, trs, brs, bls], axis = 0)
        return cboards.transpose([1,0,2])
    
    def cboard2xyhw(self, cboard):
        cboard = cboard.reshape((-1, 4, 3))
        xyz    = (cboard[:, 2, :] + cboard[:, 3, :]) / 2
        w      = np.linalg.norm(cboard[:, 2, :] - cboard[:, 3, :], axis = 1, keepdims = True)
        h      = cboard[:, 3, 2:3] - cboard[:, 0, 2:3]
        xyhw   = np.concatenate([xyz[:, :2], h, w], axis = -1)
        return xyhw

    def bbox2xyhw(self, bbox):
        cboard = self.bboxes2cboards(bbox)
        xyhw   = self.cboard2xyhw(cboard)
        return xyhw
    
    def xyhw2bbox(self, xyhw):
        cboards = self.xyhw2cboard(xyhw)
        bboxes  = self.cboards2bboxes(cboards)
        return bboxes
    
    def set_camera(self, camera):
        self.camera = camera
    
    def set_rotation(self, rotation):
        self.rotation = rotation

    def get_camera(self):
        return self.camera
    
    def get_rotation(self):
        return self.rotation

def cut_quad3d(quad3d, base_u, base_v):
    quad3d = quad3d.reshape((-1, 4, 3))
    base_u = base_u.reshape((-1, 1, 3))
    base_v = base_v.reshape((-1, 1, 3))

    u = (quad3d * base_u).sum(axis = -1)  # N x 4
    v = (quad3d * base_v).sum(axis = -1)  # N x 4
    w = quad3d[:, :, 2]                   # N x 4

    new_z = w[:, :2].max(axis = 1, keepdims = True)

    left_mean = u[:, [0, 3]].mean(axis = 1, keepdims = True)  # N x 1
    right_mean = u[:,[1, 2]].mean(axis = 1, keepdims = True)  # N x 1

    inner_mask1 = np.logical_and(u <= right_mean, u >= left_mean)
    inner_mask2 = np.logical_and(u >= right_mean, u <= left_mean)
    inner_mask  = np.logical_or(inner_mask1, inner_mask2)

    u_min, u_max = u.min(), u.max()

    new_u_min = (u * inner_mask + (1. - inner_mask) * u_max).min(axis = 1, keepdims = True)
    new_u_max = (u * inner_mask + (1. - inner_mask) * u_min).max(axis = 1, keepdims = True)
    # print(new_u_min.shape, new_u_max.shape)
    
    cboards = np.zeros(u.shape + (2,))
    cboards[:, :2, 1] = np.tile(new_z, (1, 2))
    cboards[:, 2:, 1] = 1.

    cboards[:, [0, 3], 0] = np.tile(new_u_min, (1, 2))
    cboards[:, [1, 2], 0] = np.tile(new_u_max, (1, 2))

    rect_cboards = np.zeros(quad3d.shape)
    rect_cboards[:, :, 2] = cboards[:, :, 1]
    # print(base_u[:, 0, :])
    rect_cboards[:, :, 0] = cboards[:, :, 0] * base_u[:,0,0:1] + v * base_v[:, 0,0:1]
    rect_cboards[:, :, 1] = cboards[:, :, 0] * base_u[:,0,1:2] + v * base_v[:, 0,1:2]

    return rect_cboards

# get maximum rectangle cardboard given quadrangle ones in the ground floor coordinate system
def get_max_3d_rect(cboards):
    # input shape: N x 4 x 3
    zmins   = np.min(cboards[:, :, -1], axis = -1)
    zmaxs   = np.max(cboards[:, :, -1], axis = -1)
    xmins   = np.min(cboards[:, :,  0], axis = -1)
    xmaxs   = np.max(cboards[:, :,  0], axis = -1)
    max_heights = zmaxs - zmins

    new_cardboards = cboards.copy()
    # top left
    new_cardboards[:, 0, 0]  = xmins
    new_cardboards[:, 0, -1] = 1 - max_heights

    # top right
    new_cardboards[:, 1, 0]  = xmaxs
    new_cardboards[:, 1, -1] = 1 - max_heights

    # bottom right
    new_cardboards[:, 2, 0]  = xmaxs
    new_cardboards[:, 2, -1] = np.ones(max_heights.shape)

    # bottom left
    new_cardboards[:, 3, 0]  = xmins
    new_cardboards[:, 3, -1] = np.ones(max_heights.shape)

    return new_cardboards

# get maximum rectangle box given quadrangle ones in image planes
def get_max_2d_tlwhs(quadrangles):
    xmins   = np.min(quadrangles[:, :, 0], axis = -1, keepdims = True)
    xmaxs   = np.max(quadrangles[:, :, 0], axis = -1, keepdims = True)
    ymins   = np.min(quadrangles[:, :, 1], axis = -1, keepdims = True)
    ymaxs   = np.max(quadrangles[:, :, 1], axis = -1, keepdims = True)
    widths  = xmaxs - xmins
    heights = ymaxs - ymins
    box_tlwhs = np.concatenate([xmins, ymins, widths, heights], axis = -1)
    return box_tlwhs

# convert from cboard to xyzhw
def cboard2xyzhw(cboards):
    xyzhw        = np.zeros((len(cboards), 5), dtype = np.float32)
    xyzhw[:,:3]  = (cboards[:,2,:3] + cboards[:,3,:3]) / 2   # xyz of the cardboard bottom center
    xyzhw[:, 3]  = np.abs(cboards[:,0,2] - cboards[:,3,2])   # cardboard height
    xyzhw[:, 4]  = np.abs(cboards[:,0,0] - cboards[:,1,0])   # cardboard width
    return xyzhw

def get_core_area(cboards, aspect_ratio = 1./4):
    top_mid   = cboards[:,:2,:].mean(axis = 1)
    bot_mid   = cboards[:,2:,:].mean(axis = 1)
    # print(top_mid.shape, bot_mid.shape)
    xyzhws    = cboard2xyzhw(cboards)
    # print(xyzhws.shape)
    core_area = cboards.copy()
    core_area[:,0,0] = top_mid[:,0] - xyzhws[:,3] * aspect_ratio / 2
    core_area[:,1,0] = top_mid[:,0] + xyzhws[:,3] * aspect_ratio / 2
    core_area[:,2,0] = bot_mid[:,0] + xyzhws[:,3] * aspect_ratio / 2
    core_area[:,3,0] = bot_mid[:,0] - xyzhws[:,3] * aspect_ratio / 2
    return core_area

# input N x 3 x 2
def aera_triangle(triangles):

    triangles_2nd = np.concatenate([triangles[:,2:,:].copy(), triangles[:,:2,:].copy()], axis = 1)
    sides = np.linalg.norm(triangles - triangles_2nd, axis = -1)

    s     = sides.sum(axis = 1) / 2
    a, b, c = sides[:, 0], sides[:, 1], sides[:, 2]
    # print(a, b, c)
    aeras   = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return aeras

def aera_quadrangle(quandrange_2d):
    triangle1 = quandrange_2d[:,[0, 1, 2], :]
    trianlge2 = quandrange_2d[:,[2, 3, 0], :]
    tri_aera1 = aera_triangle(triangle1)
    tri_aera2 = aera_triangle(trianlge2)
    return tri_aera1 + tri_aera2

def get_aligned_3d_rect(quadrangles):
    
    bot_xmins = np.min(quadrangles[:, 2:, 0], axis = -1)
    bot_xmaxs = np.max(quadrangles[:, 2:, 0], axis = -1)
    widths    = np.abs(bot_xmaxs - bot_xmins)
    aeras     = aera_quadrangle(quandrange_2d = quadrangles[:, :, [0, 2]])
    heights   = aeras / widths

    new_cardboards = quadrangles.copy()
    # top left
    new_cardboards[:, 0, 0]  = bot_xmins
    new_cardboards[:, 0, -1] = 1 - heights

    # top right
    new_cardboards[:, 1, 0]  = bot_xmaxs
    new_cardboards[:, 1, -1] = 1 - heights

    # bottom right
    new_cardboards[:, 2, 0]  = bot_xmaxs
    new_cardboards[:, 2, -1] = np.ones(heights.shape)

    # bottom left
    new_cardboards[:, 3, 0]  = bot_xmins
    new_cardboards[:, 3, -1] = np.ones(heights.shape)

    return new_cardboards

def get_aligned_2d_tlwhs(quadrangles):
    xmins   = np.min(quadrangles[:, 2:, 0], axis = -1, keepdims = True)
    xmaxs   = np.max(quadrangles[:, 2:, 0], axis = -1, keepdims = True)
    y       = (quadrangles[:, 2, 1] + quadrangles[:, 3, 1]) / 2
    y       = y.reshape((-1, 1))
    aeras   = aera_quadrangle(quadrangles).reshape((-1, 1))
    # print(aeras)
    widths  = xmaxs - xmins
    heights = aeras / widths
    # print(xmins.shape, y.shape, widths.shape, heights.shape)
    box_tlwhs = np.concatenate([xmins, y - heights, widths, heights], axis = -1)
    return box_tlwhs

def get_core_area(cboards, aspect_ratio = 1./4):
    top_mid   = cboards[:,:2,:].mean(axis = 1)
    bot_mid   = cboards[:,2:,:].mean(axis = 1)
    # print(top_mid.shape, bot_mid.shape)
    xyzhws    = cboard2xyzhw(cboards)
    # print(xyzhws.shape)
    core_area = cboards.copy()
    core_area[:,0,0] = top_mid[:,0] - xyzhws[:,3] * aspect_ratio / 2
    core_area[:,1,0] = top_mid[:,0] + xyzhws[:,3] * aspect_ratio / 2
    core_area[:,2,0] = bot_mid[:,0] + xyzhws[:,3] * aspect_ratio / 2
    core_area[:,3,0] = bot_mid[:,0] - xyzhws[:,3] * aspect_ratio / 2
    return core_area

# xyh -> cboards
# xyh: N x 3
# cboards: N x 4 x 3
def xyzh2rectangles3d(xyzh, aspect_ratio = 1./4, cam_height = 1.):
    
    cboard = np.zeros((len(xyzh), 4, 3))
    cboard[:, :, 1] = np.tile(xyzh[:, 1:2], [1, 4])
    
    # print(xyh, aspect_ratio)
    # print(xyzh[:, 3])
    # print(xyzh.shape)
    w = xyzh[:, -1] * aspect_ratio
    # print(w)

    # top left
    cboard[:, 0, 0] = xyzh[:, 0] - w / 2
    cboard[:, 0, 2] = cam_height - xyzh[:, -1]

    # top right
    cboard[:, 1, 0] = xyzh[:, 0] + w / 2
    cboard[:, 1, 2] = cam_height - xyzh[:, -1]

    # bottom right
    cboard[:, 2, 0] = xyzh[:, 0] + w / 2
    cboard[:, 2, 2] = cam_height

    # bottom left
    cboard[:, 3, 0] = xyzh[:, 0] - w / 2
    cboard[:, 3, 2] = cam_height

    return cboard