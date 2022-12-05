''' Tool box for visualizing the images'''
import cv2 as cv
import numpy as np
import utils

np.random.seed(0)
nb_color = 30000
color = np.random.randint(0,255,(nb_color,3))

# img: RGB, pts: N x 2
def draw_corners(img, pts, dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b),dot_size,color[i % nb_color].tolist(),-1)
    return img

def draw_corners2(img, pts, color = (0, 0, 255), dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b),dot_size,color,-1)
    return img

def draw_corners3(img, pts, id, dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        if i == 0:
            img = cv.circle(img,(a,b),dot_size,(255, 255, 255),-1)
        else:
            img = cv.circle(img,(a,b),dot_size,color[id + i].tolist(),-1)
    return img

def draw_corners_v4(img, pts, color = (0, 0, 255), dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b),dot_size,color,-1)
    return img

def draw_circle(img, pts, color = (0, 0, 255), dot_size = 8,
                thickness = 2):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b), dot_size, color, thickness)
    return img

def draw_lines(image, line_pts, color, thickness):
    for i in range(len(line_pts)):
        pt1, pt2 = np.int16(line_pts[i, :])
        # print(pt1, pt2)
        cv.line(image, tuple(pt1), tuple(pt2), color, thickness)

def draw_tracks2(image, tracks, dot_size = 1):
    for i in range(tracks.shape[0]):
        track = tracks[i, :, :]
        draw_corners2(image, track, color = color[i].tolist(), dot_size = dot_size)

def draw_tracks3(image, tracks, dot_size = 1):
    for i in range(tracks.shape[0]):
        track = tracks[i, :, :]
        if i == 0:
            draw_corners2(image, track, color = (255, 255, 255), dot_size = dot_size)
        else:
            draw_corners2(image, track, color = color[i].tolist(), dot_size = dot_size)

def draw_track_line_pts(track_pts, line_pts, size = (1920, 1080), file_name = '.'):
    gr_pts = track_pts.reshape((-1, 4, 2))
    line_pts = line_pts.reshape((-1, 2))
    w_img, h_img = size
    w_img, h_img = int(w_img), int(h_img)
    board = np.zeros((h_img, w_img, 3), dtype = np.uint8) * 255

    line_pts = np.int16(line_pts)
    rpt1, rpt2, rpt3, rpt4 = line_pts[0, :], line_pts[1, :], line_pts[2, :], line_pts[3, :]
    cv.line(board, tuple(rpt1), tuple(rpt2), (0, 255, 0), 2)
    cv.line(board, tuple(rpt3), tuple(rpt4), (0, 255, 0), 2)
    for i in range(gr_pts.shape[0]):
        track = gr_pts[i, :,:]
        draw_corners2(board, track, color = color[i].tolist(), dot_size = 3)
    cv.imwrite(file_name, board)

def draw_track_pts(track_pts, size = (1920, 1080), file_name = '.'):
    gr_pts = track_pts.reshape((-1, 4, 2))
    w_img, h_img = size
    w_img, h_img = int(w_img), int(h_img)
    board = np.zeros((h_img, w_img, 3), dtype = np.uint8) * 255

    for i in range(gr_pts.shape[0]):
        track = gr_pts[i, :,:]
        draw_corners2(board, track, color = color[i].tolist(), dot_size = 2)
    cv.imwrite(file_name, board)

def draw_box(image, box, color, thickness = 2):
    x, y, w, h = box
    pt1 = np.int16(np.array([x, y]))
    pt2 = np.int16(np.array([x + w, y]))
    pt3 = np.int16(np.array([x + w, y + h]))
    pt4 = np.int16(np.array([x, y + h]))
    # print(pt1, pt2, pt3, pt4)
    cv.line(image, pt1, pt2, color, thickness)
    cv.line(image, pt2, pt3, color, thickness)
    cv.line(image, pt3, pt4, color, thickness)
    cv.line(image, pt4, pt1, color, thickness)

def draw_quad_box(image, box, color, thickness = 2):
    # x, y, w, h = box
    # pt1 = np.int16(np.array([x, y]))
    # pt2 = np.int16(np.array([x + w, y]))
    # pt3 = np.int16(np.array([x + w, y + h]))
    # pt4 = np.int16(np.array([x, y + h]))

    pt1, pt2, pt3, pt4 = np.int16(box)
    # print(pt1, pt2, pt3, pt4)
    cv.line(image, pt1, pt2, color, thickness)
    cv.line(image, pt2, pt3, color, thickness)
    cv.line(image, pt3, pt4, color, thickness)
    cv.line(image, pt4, pt1, color, thickness)

def draw_boxes(image, boxes, color, thickness = 1):
    for i in range(len(boxes)):
        box = boxes[i,:].flatten()
        # print(box)
        draw_box(image, box, color, thickness)

def draw_text(img, text, pt, color = (255, 0, 0), scale = 10, thickness = 2):
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = int(pt[0]), int(pt[1])
    # Using cv2.putText() method
    img = cv.putText(img, text, (x, y), font, 
                    scale, color, thickness, cv.LINE_AA)
    return img

def create_draw_board(w_img, h_img):
    return np.zeros((h_img, w_img, 3), dtype = np.uint8)

def draw_pairs(file_name, pts, color, h_img):

    img_pts = pts.copy()
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin)
    tx, ty = 0, 0
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    print(img_pts)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    print(img_pts)
    img_pts = np.int16(img_pts)
    draw_board = utils.create_draw_board(w_gr, h_img)
    rpt1, rpt2, rpt3, rpt4 = img_pts[0, :], img_pts[1, :], img_pts[2, :], img_pts[3, :]
    cv.line(draw_board, tuple(rpt1), tuple(rpt2), color, 2)
    cv.line(draw_board, tuple(rpt3), tuple(rpt4), color, 2)
    cv.imwrite(file_name, draw_board)

def draw_all_pairs(file_name, pts, colors, h_img):

    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin)
    tx, ty = 0, 0
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    img_pts = np.int16(img_pts)
    img_pts = img_pts.reshape((-1, 4, 2))
    draw_board = utils.create_draw_board(w_gr, h_img)
    for i in range(img_pts.shape[0]):
        rpt1, rpt2, rpt3, rpt4 = img_pts[i, 0, :], img_pts[i, 1, :], img_pts[i, 2, :], img_pts[i, 3, :]
        cv.line(draw_board, tuple(rpt1), tuple(rpt2), colors[i].tolist(), 2)
        cv.line(draw_board, tuple(rpt3), tuple(rpt4), colors[i].tolist(), 2)
    cv.imwrite(file_name, draw_board)

def draw_all_pts(file_name, pts, color, dot_size, h_img):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin + eps)
    tx, ty = 0, 0
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    # img_pts = np.int16(img_pts)
    # img_pts = img_pts.reshape((-1, 4, 2))
    draw_board = utils.create_draw_board(w_gr, h_img)
    draw_corners2(draw_board, img_pts, color, dot_size)
    # for i in range(img_pts.shape[0]):
    #     rpt1, rpt2, rpt3, rpt4 = img_pts[i, 0, :], img_pts[i, 1, :], img_pts[i, 2, :], img_pts[i, 3, :]
    #     cv.line(draw_board, tuple(rpt1), tuple(rpt2), colors[i].tolist(), 2)
    #     cv.line(draw_board, tuple(rpt3), tuple(rpt4), colors[i].tolist(), 2)
    cv.imwrite(file_name, draw_board)

def draw_pts_on_floor(file_name, pts, colors, dot_size, h_img):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin + eps) * 0.8
    tx, ty = 10, 10
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = utils.create_draw_board(w_gr, h_img)
    for i in range(pts.shape[0]):
        if i == 0: draw_corners2(draw_board, img_pts[i: i + 1, :], (255, 255, 255), dot_size)
        else: draw_corners2(draw_board, img_pts[i: i + 1, :], colors[i].tolist(), dot_size)
    cv.imwrite(file_name, draw_board)

def draw_pts_on_floor_v3(file_name, pts, heights, colors, dot_size, h_img, scale = 1, thickness = 1):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin + eps) * 0.8
    tx, ty = 10, 10
    min_w = 200
    w_gr = max(int((xmax - xmin) * scaling_factor + tx * 2), min_w)
    # print(w_gr)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = utils.create_draw_board(w_gr, h_img)
    
    for i in range(pts.shape[0]):
        if i == 0: draw_corners2(draw_board, img_pts[i: i + 1, :], (255, 255, 255), dot_size)
        else: draw_corners2(draw_board, img_pts[i: i + 1, :], colors[i].tolist(), dot_size)
        # print(heights[i])
        draw_text(draw_board, '{:.2f}'.format(heights[i]), img_pts[i, :], color = colors[i].tolist(), scale = 1, thickness = 1)
    cv.imwrite(file_name, draw_board)

def draw_pts_on_floor_v4(pts, heights, colors, dot_size, h_img, scale = 1, thickness = 1):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin + eps) * 0.8
    tx, ty = 10, 10
    min_w = 200
    w_gr = max(int((xmax - xmin) * scaling_factor + tx * 2), min_w)
    # print(w_gr)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = utils.create_draw_board(w_gr, h_img)
    
    for i in range(pts.shape[0]):
        if i == 0: draw_corners2(draw_board, img_pts[i: i + 1, :], (255, 255, 255), dot_size)
        else: draw_corners2(draw_board, img_pts[i: i + 1, :], colors[i].tolist(), dot_size)
        # print(heights[i])
        draw_text(draw_board, '{:.2f}'.format(heights[i]), img_pts[i, :], color = colors[i].tolist(), scale = 1, thickness = 1)
    
    return draw_board

def draw_errs_on_floor(pts, trk_ids, colors, dot_size, h_img, scale = 1, thickness = 1):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    scaling_factor = h_img / (ymax - ymin + eps) * 0.8
    tx, ty = 10, 10
    min_w  = 200
    w_gr   = max(int((xmax - xmin) * scaling_factor + tx * 2), min_w)

    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = utils.create_draw_board(w_gr, h_img)
    
    for i in range(pts.shape[0]):
        if i == 0: draw_corners2(draw_board, img_pts[i: i + 1, :], (255, 255, 255), dot_size)
        else: draw_corners2(draw_board, img_pts[i: i + 1, :], colors[i].tolist(), dot_size)

        draw_text(draw_board, '{}'.format(int(trk_ids[i])), img_pts[i, :], color = colors[i].tolist(), scale = scale, 
                  thickness = thickness)
    
    return draw_board

def draw_errs_on_floor_v2(gt_xys, trk_xys, matches, fns, fps, switched_ids, 
                          gt_colors, trk_colors, xy_range, colors, dot_size, 
                          h_img, scale = 1, thickness = 1, circle_tk = 2,
                          orient_angle = 90, r = 20, arrow_thickness = 2):
    eps = 1e-8
    gt_pts  = np.array([gt_xys[id] for id in gt_xys]).reshape((-1, 2))
    trk_pts = np.array([trk_xys[id] for id in trk_xys]).reshape((-1, 2))
    img_pts = np.concatenate([gt_pts, trk_pts], axis = 0)
    img_pts = img_pts.reshape((-1, 2))
    # xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    xmin, xmax, ymin, ymax = xy_range

    # include camera location
    xmin, xmax = min(xmin, 0), max(xmax, 0)
    ymin, ymax = min(ymin, 0), max(ymax, 0)
    scaling_factor = h_img / (ymax - ymin + eps) * 0.95
    tx, ty = 10, 10
    min_w  = 200
    w_gr   = max(int((xmax - xmin) * scaling_factor + tx * 2), min_w)

    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    # draw_board = utils.create_draw_board(w_gr, h_img)
    draw_board = utils.create_white_board(w_gr, h_img)
    
    # build trk to gt match directory
    match_dic = {}
    for m in matches:
        if not m[0] in match_dic: match_dic[m[0]] = m[1]
    
    # draw true postives
    for m in matches:
        gt_id, trk_id = m
        x, y = gt_xys[gt_id].flatten()
        x = (x - xmin) * scaling_factor + tx
        y = (y - ymin) * scaling_factor + ty
        gt_xy = np.array([x, y]).reshape((-1, 2))

        # if not gt_id in gt_colors: gt_colors[gt_id] = gt_id
        # elif trk_id in switched_ids: gt_colors[gt_id] = gt_id
        # used_color = colors[gt_colors[gt_id]].tolist()

        # used_color = colors[trk_id].tolist()
        used_color = colors[gt_id].tolist()

        draw_corners2(draw_board, gt_xy, used_color, dot_size)
        # draw_text(draw_board, '{}'.format(gt_id), gt_xy[0, :], color = used_color, scale = scale, 
        #           thickness = thickness)

        # draw matches
        x_trk, y_trk = trk_xys[trk_id].flatten()
        x_trk    = (x_trk - xmin) * scaling_factor + tx
        y_trk    = (y_trk - ymin) * scaling_factor + ty
        line_pts = np.array([x, y, x_trk, y_trk]).reshape((-1, 2, 2))
        draw_lines(draw_board, line_pts, used_color, thickness)

        # draw switched ids
        if trk_id in switched_ids:
            # print('switch id: {}'.format(trk_id))
            draw_text(draw_board, '{}'.format(trk_id), gt_xy[0, :], color = used_color, scale = scale, 
                      thickness = thickness)


    # draw false negatives
    for fn_gt_id in fns:
        # gt_xy = np.array(gt_xys[fn_gt_id]).reshape((-1, 2))
        x, y = gt_xys[fn_gt_id].flatten()
        x = (x - xmin) * scaling_factor + tx
        y = (y - ymin) * scaling_factor + ty
        gt_xy = np.array([x, y]).reshape((-1, 2))
        # draw_circle(draw_board, gt_xy, colors[fn_gt_id].tolist(), dot_size,
        #             thickness = circle_tk)
        # draw_circle(draw_board, gt_xy, (0, 0, 0), dot_size,
        #             thickness = circle_tk)
        draw_circle(draw_board, gt_xy, colors[fn_gt_id].tolist(), dot_size,
                    thickness = circle_tk)
        # draw_text(draw_board, '{}'.format(fn_gt_id), gt_xy[0,:], color = (0, 0, 0), scale = scale, 
        #           thickness = thickness)

    # draw false positives
    for fp_trk_id in fps:
        # trk_xy = np.array(trk_xys[fp_trk_id]).reshape((-1, 2))
        x, y = trk_xys[fp_trk_id].flatten()
        x = (x - xmin) * scaling_factor + tx
        y = (y - ymin) * scaling_factor + ty
        trk_xy = np.array([x, y]).reshape((-1, 2))
        # draw_circle(draw_board, trk_xy, colors[fp_trk_id].tolist(), dot_size,
        #             thickness = circle_tk)
        draw_circle(draw_board, trk_xy, (0, 0, 0), dot_size,
                    thickness = circle_tk)
        draw_text(draw_board, '{}'.format(fp_trk_id), trk_xy[0,:], color = (0, 0, 0), scale = scale, 
                  thickness = thickness)

    cx = int((0 - xmin) * scaling_factor + tx)
    cy = int((0 - ymin) * scaling_factor + ty)

    # print(cx, cy)
    orient_angle = np.pi * orient_angle / 180
    draw_corners_v4(draw_board, np.array([cx, cy]).reshape((-1, 2)), (0, 0, 255), dot_size + 4)
    dx, dy = r * np.cos(orient_angle), r * np.sin(orient_angle)
    cv.arrowedLine(draw_board, (cx, cy), (int(cx + dx), int(cy + dy)),
                    color = (0, 0, 255), thickness = arrow_thickness, tipLength = 0.6)
    
    return draw_board

def draw_pts_in_images(file_name, pts, colors, dot_size, w_img, h_img):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    draw_board = utils.create_draw_board(w_img, h_img)
    for i in range(pts.shape[0]):
        if i == 0: draw_corners2(draw_board, img_pts[i: i + 1, :], (0, 0, 0), dot_size)
        else: draw_corners2(draw_board, img_pts[i: i + 1, :], colors[i].tolist(), dot_size)
    cv.imwrite(file_name, draw_board)

def draw_pairs_in_images(draw_board, pairs, colors, dot_size):
    for i in range(pairs.shape[0]):
        pair   = pairs[i, :, :]
        first  = pair[:2, :]
        second = pair[2:, :]
        draw_corners2(draw_board, first, colors[i].tolist(), dot_size)
        draw_corners2(draw_board, second, colors[i].tolist(), dot_size)


def draw_pts_on_floor_v2(file_name, pts, colors, ids, dot_size, h_img):
    eps = 1e-8
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    xmin, xmax = min(xmin, 0), max(xmax, 0)
    ymin, ymax = min(ymin, 0), max(ymax, 0)
    scaling_factor = h_img / (ymax - ymin + eps) * 0.8
    tx, ty = 10, 10
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    # print(img_pts)
    draw_board = utils.create_draw_board(w_gr, h_img)
    for i in range(pts.shape[0]):
        # print(i, ids[i])
        draw_corners2(draw_board, img_pts[i: i + 1, :], colors[ids[i]].tolist(), dot_size)
    cx = int((0 - xmin) * scaling_factor + tx * 2)
    cy = int((0 - ymin) * scaling_factor + ty * 2)
    draw_corners2(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size + 4)
    cv.imwrite(file_name, draw_board)