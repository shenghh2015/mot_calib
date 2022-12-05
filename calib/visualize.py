from utils import get_box_centers
import cv2 as cv
import numpy as np

from const import RANDOM_COLORS, EPS

def create_white_board(w_img, h_img):
    return np.ones((h_img, w_img, 3), dtype = np.uint8) * 128

def create_bordered_white_board(w_img, h_img):
    board = np.ones((h_img, w_img, 3), dtype = np.uint8) * 128
    board[0, :, :] = 255
    board[h_img - 1, :, :] = 255
    board[:, 0, :] = 255
    board[:, w_img - 1, :] = 255
    return board

def create_black_board(w_img, h_img):
    return np.zeros((h_img, w_img, 3), dtype = np.uint8)

def draw_corners(img, pts, color = (0, 0, 255), dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b),dot_size,color,-1)
    return img

def draw_text(img, text, pt, color = (255, 0, 0), scale = 10, thickness = 2):
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = int(pt[0]), int(pt[1])
    img = cv.putText(img, text, (x, y), font, 
                    scale, color, thickness, cv.LINE_AA)

def vis_xyz_as_images(xyh, track_ids, rngs, img_size, dot_size = 2, 
                        scale = 1, thickness = 2, orient_angle = 90, 
                        r = 10, arrow_thickness = 2, colors = []):
    xyh_copy    = xyh.copy()
    img_pts     = xyh_copy[:, :2]
    heights     = xyh_copy[:, 2].flatten()
    # colors      = RANDOM_COLORS
    # if use_flip:
    #     img_pts[:, 1] = - img_pts[:, 1]
    # xmax, ymax = rngs[: 2]
    # xmin, ymin = - xmax, - ymax * 0.1
    xmin, ymin, xmax, ymax = rngs
    xmin = min(xmin, 0)
    ymin = min(ymin, 0)
    ymax = max(0, ymax)
    core_ratio = 0.9
    scaling_factor = img_size / (ymax - ymin + EPS) * core_ratio
    # scaling_factor2 = img_size / (xmax - xmin + EPS) * core_ratio
    # scaling_factor = min(scaling_factor1, scaling_factor2)

    tx, ty = img_size // 2 + 20, 20
    xmin = 0
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = create_white_board(img_size, img_size)
    for i in range(img_pts.shape[0]):
        # print(i, len(track_ids))
        draw_corners(draw_board, img_pts[i: i + 1, :], colors[track_ids[i]].tolist(), dot_size)
        # print(heights)
        # print(heights[i][0])
        draw_text(draw_board, '{:.2f}'.format(heights[i]), img_pts[i, :] + 2, color = colors[track_ids[i]].tolist(), scale = scale, thickness = thickness)
    cx = int((0 - xmin) * scaling_factor + tx)
    cy = int((0 - ymin) * scaling_factor + ty)

    draw_corners(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size)
    # dx, dy = r * np.cos(orient_angle), r * np.sin(orient_angle)
    # cv.arrowedLine(draw_board, (cx, cy), (int(cx + dx), int(cy + dy)),
    #                 color = (255, 255, 255), thickness = arrow_thickness, tipLength = 0.5)
    return draw_board

def vis_xyz_as_images_v2(xyh, track_ids, rngs, img_size, dot_size = 2, 
                        scale = 1, thickness = 2, orient_angle = 90, 
                        r = 10, arrow_thickness = 2, colors = []):
    
    img_pts     = xyh[:, :2]
    heights     = xyh[:, 2].flatten()
    # colors      = RANDOM_COLORS
    # if use_flip:
    #     img_pts[:, 1] = - img_pts[:, 1]
    # xmax, ymax = rngs[: 2]
    # xmin, ymin = - xmax, - ymax * 0.1
    xmin, ymin, xmax, ymax = rngs
    xmin = min(xmin, 0)
    xmax = max(xmax, 0)
    ymin = min(ymin, 0)
    ymax = max(0, ymax)
    core_ratio = 0.9
    scaling_factor1 = img_size / (ymax - ymin + EPS) * core_ratio
    scaling_factor2 = img_size / (xmax - xmin + EPS) * core_ratio
    scaling_factor  = min(scaling_factor1, scaling_factor2)

    tx, ty = img_size // 2 + 20, 20
    xmin = 0
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = create_white_board(img_size, img_size)
    for i in range(img_pts.shape[0]):
        # print(i, len(track_ids))
        draw_corners(draw_board, img_pts[i: i + 1, :], colors[track_ids[i]].tolist(), dot_size)
        # print(heights)
        # print(heights[i][0])
        draw_text(draw_board, '{:.2f}'.format(heights[i]), img_pts[i, :] + 2, color = colors[track_ids[i]].tolist(), scale = scale, thickness = thickness)
    cx = int((0 - xmin) * scaling_factor + tx)
    cy = int((0 - ymin) * scaling_factor + ty)

    draw_corners(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size)
    # dx, dy = r * np.cos(orient_angle), r * np.sin(orient_angle)
    # cv.arrowedLine(draw_board, (cx, cy), (int(cx + dx), int(cy + dy)),
    #                 color = (255, 255, 255), thickness = arrow_thickness, tipLength = 0.5)
    return draw_board

def plot_2d_pts_as_imgs(xyh, val_rng = [1, 1], labels = ['x', 'y'], 
                        fig_size = 5, title = ''):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = 1, 1, fig_size
    font_size = 15
    fig = Figure(tight_layout=True, figsize=(size*cols, size*rows)); 
    ax = fig.subplots(rows,cols)
    ax.scatter(xyh[:, 0], xyh[:, 1], color = 'g')
    ax.set_xlabel(labels[0], fontsize = font_size)
    ax.set_ylabel(labels[1], fontsize = font_size)
    ax.set_xlim([-val_rng[0], val_rng[0]])
    ax.set_ylim([-val_rng[1], val_rng[1]])
    ax.set_title(title)
    fig_img = get_img_from_fig(fig, dpi = 100)

    return fig_img

def plot_height_stats(heights, labels = ['x', 'y'], fig_size = 5, title = ''):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows, cols, size = 1, 1, fig_size
    font_size = 15
    fig = Figure(tight_layout=True, figsize=(size*cols, size*rows)); 
    ax = fig.subplots(rows,cols)
    ax.hist(heights, bins = 100)
    ax.set_xlabel(labels[0], fontsize = font_size)
    ax.set_ylabel(labels[1], fontsize = font_size)
    ax.set_title(title)
    fig_img = get_img_from_fig(fig, dpi = 100)

    return fig_img

import io
# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img

def plot_loss_1d(file_name, losses, xlist, 
                 title = '', labels = ['fov', 'height loss'], fig_size = 5,
                 use_clip = True, clip_x = []):

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = 1, 1, fig_size
    font_size = 15
    fig = Figure(tight_layout=True, figsize=(size*cols, size*rows)); 
    ax = fig.subplots(rows,cols)
    if use_clip:
        losses = np.clip(losses, 0, 1.0)
    if clip_x:
        xarr = np.array(xlist)
        larr = np.array(losses)
        ind_mask = np.where(np.logical_and(xarr > clip_x[0], xarr < clip_x[1]))[0]
        # print(ind_mask)
        xarr = xarr[ind_mask]
        larr = larr[ind_mask]
        ax.plot(xarr, larr)
    else:
        ax.plot(xlist, losses)
    ax.set_xlabel(labels[0], fontsize = font_size)
    ax.set_ylabel(labels[1], fontsize = font_size)
    ax.set_title(title)
    # fig_img = get_img_from_fig(fig, dpi = 100)

    canvas = FigureCanvasAgg(fig); 
    canvas.print_figure(file_name, dpi=100)

def vis_box_pairs(w_img, h_img, box_pairs):

    board = create_white_board(w_img, h_img)

    for i in range(box_pairs.shape[0]):

        boxes       = box_pairs[i, :].reshape((-1, 4))
        bot_centers = get_box_centers(boxes)
        bot_centers = np.int16(bot_centers)
        color = RANDOM_COLORS[i].tolist()
        thickness = 2
        draw_corners(board, bot_centers, color, dot_size = 5)
        cv.line(board, bot_centers[0,:], bot_centers[1,:], color, thickness)

    return board