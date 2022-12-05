import yolox.camera.geometry_3d as gm3d
import numpy as np

class CardBoard:

    def __init__(self, camera, surf_normal, cam_height = 1.):

        self.camera      = camera
        self.surf_normal = surf_normal
        self.cam_height  = cam_height

        # Initialize rotation object for translation between the camera coordinates and
        #  averial view coordinates
        rot_matrix       = self._gen_rotation_matrix(surf_normal)
        self.rotation    = gm3d.Rotation()
        self.rotation.set_R(rot_matrix)
    
    # Generate the rotation matrix based on the surface normal
    def _gen_rotation_matrix(self, surf_normal):

        cam_z_base  = np.array([0, 0, 1.], dtype = np.float32) # z base vector in camera coordinate sytem
        x_base = np.cross(cam_z_base, surf_normal)
        y_base = cam_z_base - np.sum(cam_z_base * surf_normal) * surf_normal
        x_base = x_base / np.linalg.norm(x_base)
        y_base = y_base / np.linalg.norm(y_base)
        return np.stack([x_base, y_base, surf_normal], axis = 0).reshape((3, 3))

    def set_rot_matrix(self, rot_matrix):
        self.rotation.set_R(rot_matrix)
    
    # reject the bboxes that are above vanishing line
    # return the indices of boxes that are below the vanishing line
    def reject_bboxes_above_horizon(self, bboxes):

        # botteom cetners of bboxes in image plane
        bcs = bboxes[:, :2].copy()
        bcs[:, 0] += bboxes[:, 2] / 2
        bcs[:, 1] += bboxes[:, 3]

        # 3d rays in camera coordinate system
        bc_rays = self.camera.map_2d_pts_to_3d_rays(bcs).T
        
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

    def bboxes_to_quadrangles3d(self, bboxes):

        # top left, top right, bottom right, bottom left, bottom centers: N x 2
        tls  = bboxes[:, :2]                  
        brs  = bboxes[:, :2] + bboxes[:, 2:]

        trs  = tls.copy()
        trs[:, 0] += bboxes[:, 2]

        bls  = tls.copy()
        bls[:, 1] += bboxes[:, 3]

        bcs  = (bls + brs) / 2     # bottom center

        # 3d rays related to top left, top right, bottom right, bottom left, bottom center of boxes
        pts_2d        = np.concatenate([tls, trs, brs, bls, bcs], axis = 0)  # 5 x N x 3
        rays          = self.camera.map_2d_pts_to_3d_rays(pts_2d).T          # 5 x N x 3
        rays          = self.trans_to_aerial(rays)
        
        corner_rays   = rays[:4 *len(tls), :]     # rays related to corners of bboxes
        bc_rays       = rays[4*len(tls):,  :]     # rays related to bottom centers of bboxes
        grf_bcs       = gm3d.normalize_rays_3d(bc_rays.T, axis = 2).T  # projections of bottom centers on the ground floor

        # projections of corner rays to a vertical plane y = y0
        # y = y0 is determined by grf_bcs
        scales        = np.repeat(grf_bcs[:, 1:2], repeats = 4, axis = 1).T.reshape((-1, 1)) / corner_rays[:, 1:2]
        pts_3d        = scales * corner_rays

        # generate rectangle cardboards
        quadrangle_cboards = pts_3d.reshape(4, len(bboxes), 3)
        quadrangle_cboards = np.transpose(quadrangle_cboards, axes = [1, 0, 2])

        return quadrangle_cboards
    
    def cboards_to_quadrangles2d(self, cardboards):

        # translate the cardboard coordinates back to the camera coordinate system
        pts_3d       = cardboards.copy().reshape((-1, 3))
        pts_3d       = self.trans_to_camera(pts_3d)
        pts_2d       = self.camera.map_3d_rays_to_2d_pts(pts_3d.T)

        # generate rectangle bounding boxes
        quadrangle_bboxes = pts_2d.reshape((-1, 4, 2))

        return quadrangle_bboxes

    # map boxes in image plane to cardboard in aerial view
    # inputs:  bboxes N x 4 (tlwh)
    # outputs: cardboards N x 4 x 3
    def bboxes2cboards(self, bboxes, rule = 'mid_rect', use_core = False, aspect_ratio = 1./4):
        quadrangle_cboards = self.bboxes_to_quadrangles3d(bboxes)

        if rule == 'max_rect':
            rect_cardboards    = get_max_3d_rect(quadrangle_cboards) * self.cam_height
        elif rule == 'mid_rect':
            rect_cardboards    = get_aligned_3d_rect(quadrangle_cboards) * self.cam_height
        # print('cardbpard', rect_cardboards.shape)

        if use_core:
            rect_cardboards = get_core_area(rect_cardboards, aspect_ratio = aspect_ratio)

        return rect_cardboards

    # compute heights of persons in reference to a 3d plane determined by 
    # a x + b y + c z + d = 0 (c > 0, d = -1)
    # [a, b, c] = self.surf_normal
    def bboxes2rectangles3d(self, bboxes, aspect_ratio = 1./4):

        # top left, top right, bottom right, bottom left, bottom centers: N x 2
        tls  = bboxes[:, :2]                  
        brs  = bboxes[:, :2] + bboxes[:, 2:]

        trs  = tls.copy()
        trs[:, 0] += bboxes[:, 2]

        bls  = tls.copy()
        bls[:, 1] += bboxes[:, 3]

        bcs  = (bls + brs) / 2     # bottom center

        # 3d rays related to top left, top right, bottom right, bottom left, bottom center of boxes
        pts_2d  = np.concatenate([tls, trs, bcs], axis = 0)      # 3N x 3
        rays    = self.camera.map_2d_pts_to_3d_rays(pts_2d).T    # 3N x 3
        
        tl_rays  = rays[:len(tls),   :]                          # rays related to top left corners of bboxes
        tr_rays  = rays[len(tls):2*len(tls),:]                   # rays related to top right corners of bboxes
        bc_rays  = rays[2*len(tls):, :]                          # rays related to bottom centers of bboxe

        # compute heights
        heights = gm3d.get_vertical_vec_heights(bc_rays, tl_rays, tr_rays, self.surf_normal).reshape((-1, 1))

        # compute the projection of bbox bottom center
        bc_rays    = self.trans_to_aerial(bc_rays)
        grf_bcs  = gm3d.normalize_rays_3d(bc_rays.T, axis = 2).T  # projections of bottom centers on the ground floor


        xyzh     = np.concatenate([grf_bcs, heights], axis = -1)

        # xyh -> rectangles
        rect_3d = xyzh2rectangles3d(xyzh, aspect_ratio = aspect_ratio, cam_height = self.cam_height)

        return rect_3d

    # generate a cardboard based on the projection of the bbox bottom center and height
    def bboxes2cboards_v2(self, bboxes, aspect_ratio = 1./4):

        return self.bboxes2rectangles3d(bboxes, aspect_ratio = aspect_ratio)

    # map cardboard in the aerial view to the bboxes in the image plane
    # input: cardboards N x 4 x 3 
    # output: boxes N x 4 (tlwh)
    def cboards2bboxes(self, cardboards, rule = 'mid_rect'):
        quadrangle_bboxes = self.cboards_to_quadrangles2d(cardboards)

        if rule == 'max_rect':
            boxes  = get_max_2d_tlwhs(quadrangle_bboxes)
        elif rule == 'mid_rect':
            boxes  = get_aligned_2d_tlwhs(quadrangle_bboxes)
        # print('proj bbox', boxes.shape)
        return boxes

    # cboards: N x 4 x 3
    def cboards2bboxes_v2(self, cboards, aspect_ratios = np.array([])):

        # print(cboards)
        cboards_bc       = cboards[:, 3, :].copy()              # N x 3
        cboards_bc[:, 0] = (cboards[:, 3, 0] + cboards[:, 2, 0]) / 2

        cboards_tc       = cboards[:,  0, :].copy()              # N x 3
        cboards_tc[:, 0] = (cboards[:, 0, 0] + cboards[:, 1, 0]) / 2

        # top left, top right, bottom centers
        # top_two   = np.transpose(cboards[:, :2, :].copy(), [1, 0, 2])
        # cboards_tc = np.expand_dims(cboards_tc, axis = 0)
        # cboards_bc = np.expand_dims(cboards_bc, axis = 0)
        # pts_3d  = np.concatenate([top_two, cboards_bc], axis = 0)
        pts_3d  = np.stack([cboards_tc, cboards_bc], axis = 0)

        # print(pts_3d.shape)
        pts_3d  = pts_3d.reshape((-1, 3))

        # print(pts_3d)
        pts_3d  = self.trans_to_camera(pts_3d)
        # print(pts_3d.shape, pts_3d)
        pts_2d  = self.camera.map_3d_rays_to_2d_pts(pts_3d.T)
        # print(pts_2d)

        pts_2d = pts_2d.reshape((2, -1, 2))
        pts_2d = np.transpose(pts_2d, [1, 0, 2])
        ymin, ymax = pts_2d[:, :, 1].min(axis = -1), pts_2d[:, :, 1].max(axis = -1)
        h      = (ymax - ymin).reshape((-1, 1))
        # print(h)
        aspect_ratios = aspect_ratios.reshape(h.shape)
        w      = h * aspect_ratios
        tl_pts = pts_2d[:, 1, :].copy()
        tl_pts[:, 0:1] -= w / 2
        tl_pts[:, 1:2] -= h

        tlwh   = np.concatenate([tl_pts, w, h], axis = -1)
        return tlwh
    
    # translate from camera coordinate system to the aerial-view coordinate system
    def trans_to_aerial(self, pts_3d):
        new_pts_3d = self.rotation.rotate(pts_3d.T)
        return new_pts_3d.T

    # translate from the aerial-view coordinate system to camera coordinate system
    def trans_to_camera(self, pts_3d):
        new_pts_3d = self.rotation.rotate_inverse(pts_3d.T)
        return new_pts_3d.T
    
    def map_img_pts_to_grf_pts(self, box_pts):
        box_pts = box_pts.reshape((-1, 2))
        ray_3d  = self.camera.map_2d_pts_to_3d_rays(box_pts).T
        ray_3d  = self.trans_to_aerial(ray_3d)
        xyz     = gm3d.normalize_rays_3d(ray_3d.T, axis = 2).T

        # print(xyz.shape)
        return xyz[:, :2]

    # map a bbox to a 3d cardboard
    # bbox shape: 4 (tlwh)
    # cardboard shape: 4 x 3
    def bbox2cboard(self, bbox, use_core = False, aspect_ratio = 1./4, use_height = False):
        bbox = bbox.reshape((-1, 4))
        if not use_height:
            return self.bboxes2cboards(bbox, use_core = use_core, 
                                    aspect_ratio = aspect_ratio).squeeze()
        else:
            return self.bboxes2cboards_v2(bbox, aspect_ratio = aspect_ratio).squeeze()
    
    # map a 3d cardboard to a bbox
    # cardboard shape: 4 x 3
    # bbox shape: 4 (tlwh)
    def cboard2bbox(self, cboard, aspect_ratio = 1./4, use_height = False):
        cboard = cboard.reshape((-1, 4, 3))
        if not use_height:
            return self.cboards2bboxes(cboard).squeeze()
        else:
            aspect_ratios = np.array([aspect_ratio]).reshape((-1, 1))
            tlwh = self.cboards2bboxes_v2(cboard, aspect_ratios = aspect_ratios).squeeze()
            return tlwh

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

    # print(sides.shape)
    s     = sides.sum(axis = 1) / 2
    a, b, c = sides[:, 0], sides[:, 1], sides[:, 2]
    # print(a, b, c)
    aeras   = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return aeras

def aera_quadrangle(quandrange_2d):
    # print(quandrange_2d.shape)
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