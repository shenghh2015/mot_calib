# Author: Shenghua He (shenghh2015@gmail.com)
import cv2 as cv
import numpy as np

from . import geometry_3d as geom

EPS = 1e-8

def create_draw_board(w_img, h_img):
    return np.zeros((h_img, w_img, 3), dtype = np.uint8)

def draw_corners(img, pts, color = (0, 0, 255), dot_size = 8):
    for i, pt in enumerate(pts):
        a, b = pt.ravel()
        a, b = int(a), int(b)
        img = cv.circle(img,(a,b),dot_size,color,-1)
    return img

def draw_box(image, box, color, thickness = 2):
    x, y, w, h = box
    pt1 = np.int16(np.array([x, y]))
    pt2 = np.int16(np.array([x + w, y]))
    pt3 = np.int16(np.array([x + w, y + h]))
    pt4 = np.int16(np.array([x, y + h]))
    # print(image.shape, pt1, pt2, color, thickness)
    cv.line(image, pt1, pt2, color, thickness)
    cv.line(image, pt2, pt3, color, thickness)
    cv.line(image, pt3, pt4, color, thickness)
    cv.line(image, pt4, pt1, color, thickness)

def draw_boxes(image, boxes, color, thickness = 1):
    for i in range(len(boxes)):
        box = boxes[i,:].flatten()
        draw_box(image, box, color, thickness)


def draw_box_v2(image, box, color, thickness = 2):
    x, y, w, h = box
    box = np.int16(box)
    pt1 = box[0, :]
    pt2 = box[1, :]
    pt3 = box[2, :]
    pt4 = box[3, :]
    cv.line(image, pt1, pt2, color, thickness)
    cv.line(image, pt2, pt3, color, thickness)
    cv.line(image, pt3, pt4, color, thickness)
    cv.line(image, pt4, pt1, color, thickness)

def draw_boxes_v2(image, boxes, color, thickness = 1):
    for i in range(len(boxes)):
        box = boxes[i, :, :]
        draw_box_v2(image, box, color, thickness)

def draw_text(img, text, pt, color = (255, 0, 0), scale = 10, thickness = 2):
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = int(pt[0]), int(pt[1])
    img = cv.putText(img, text, (x, y), font, 
                    scale, color, thickness, cv.LINE_AA)

def draw_pts_in_images(image, pts, colors, ids, dot_size, w_img, h_img):
    img_pts = pts.reshape((-1, 2))
    for i in range(pts.shape[0]):
        draw_corners(image, img_pts[i: i + 1, :], colors[ids[i]].tolist(), dot_size)

def draw_pts_on_floor(pts, colors, ids, dot_size, h_img, use_flip = False, rot_angle = 0):
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    if use_flip:
        img_pts[:, 1] = - img_pts[:, 1]
    if not rot_angle == 0:
        theta = rot_angle * np.pi / 180
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        img_pts = geom.rotation_2d(R, img_pts)
    xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    xmin, xmax = min(xmin, 0), max(xmax, 0)
    ymin, ymax = min(ymin, 0), max(ymax, 0)
    scaling_factor = h_img / (ymax - ymin + EPS) * 0.9
    tx, ty = 20, 20
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = create_draw_board(w_gr, h_img)
    for i in range(pts.shape[0]):
        draw_corners(draw_board, img_pts[i: i + 1, :], colors[ids[i]].tolist(), dot_size)
    cx = int((0 - xmin) * scaling_factor + tx * 2)
    cy = int((0 - ymin) * scaling_factor + ty * 2)
    draw_corners(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size)
    return draw_board

# specify the cor_ranges before call this function
# xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
# xmin, xmax = min(xmin, 0), max(xmax, 0)
# ymin, ymax = min(ymin, 0), max(ymax, 0)
# cor_ranges = np.array([xmin, xmax, ymin, ymax])
def draw_pts_on_floor2(pts, colors, ids, dot_size, h_img, cor_ranges, use_flip = False, rot_angle = 0):
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    if use_flip:
        img_pts[:, 1] = - img_pts[:, 1]
    if not rot_angle == 0:
        theta = rot_angle * np.pi / 180
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        img_pts = geom.rotation_2d(R, img_pts)
    # xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
    # xmin, xmax = min(xmin, 0), max(xmax, 0)
    # ymin, ymax = min(ymin, 0), max(ymax, 0)
    xmin, xmax, ymin, ymax = cor_ranges.flatten()
    scaling_factor = h_img / (ymax - ymin + EPS) * 0.9
    tx, ty = 20, 20
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = create_draw_board(w_gr, h_img)
    for i in range(pts.shape[0]):
        draw_corners(draw_board, img_pts[i: i + 1, :], colors[ids[i]].tolist(), dot_size)
    cx = int((0 - xmin) * scaling_factor + tx * 2)
    cy = int((0 - ymin) * scaling_factor + ty * 2)
    draw_corners(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size)
    return draw_board

# specify the cor_ranges before call this function
# xmin, xmax, ymin, ymax = img_pts[:, 0].min(), img_pts[:, 0].max(), img_pts[:, 1].min(), img_pts[:, 1].max()
# xmin, xmax = min(xmin, 0), max(xmax, 0)
# ymin, ymax = min(ymin, 0), max(ymax, 0)
# cor_ranges = np.array([xmin, xmax, ymin, ymax])

# plot points on the ground floor and show the heights
def draw_pts_on_floor3(pts, heights, colors, ids, dot_size, h_img, cor_ranges,
                       scale = 1, thickness = 2,
                       use_flip = False, rot_angle = 0,
                       orient_angle = 90, r = 10, arrow_thickness = 2):
    img_pts = pts.copy()
    img_pts = img_pts.reshape((-1, 2))
    if use_flip:
        img_pts[:, 1] = - img_pts[:, 1]
    if not rot_angle == 0:
        theta = rot_angle * np.pi / 180
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        img_pts = geom.rotation_2d(R, img_pts)
    xmin, xmax, ymin, ymax = cor_ranges.flatten()
    scaling_factor = h_img / (ymax - ymin + EPS) * 0.9
    tx, ty = 20, 20
    w_gr = int((xmax - xmin) * scaling_factor + tx * 2)
    img_pts[:, 0] = (img_pts[:, 0] - xmin) * scaling_factor + tx
    img_pts[:, 1] = (img_pts[:, 1] - ymin) * scaling_factor + ty
    draw_board = create_draw_board(w_gr, h_img)
    for i in range(pts.shape[0]):
        draw_corners(draw_board, img_pts[i: i + 1, :], colors[ids[i]].tolist(), dot_size)
        draw_text(draw_board, '{:.2f}'.format(heights[i]), img_pts[i, :] + 2, color = colors[ids[i]].tolist(), scale = scale, thickness = thickness)
    cx = int((0 - xmin) * scaling_factor + tx * 2)
    cy = int((0 - ymin) * scaling_factor + ty * 2)
    draw_corners(draw_board, np.array([cx, cy]).reshape((-1, 2)), (255, 255, 255), dot_size)
    dx, dy = r * np.cos(orient_angle), r * np.sin(orient_angle)
    cv.arrowedLine(draw_board, (cx, cy), (int(cx + dx), int(cy + dy)),
                    color = (255, 255, 255), thickness = arrow_thickness, tipLength = 0.5)
    return draw_board


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv.line(img,s,e,color.tolist(),thickness)
            i+=1

def drawpoly(img, pts,color,thickness=1,style='dotted', gap = 40):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style, gap = gap)

def drawrect(img,pt1,pt2,color,thickness=1, style='dotted', gap = 40):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style, gap = gap)

def draw_rect_box(img, box, color, thickness = 1, 
                  style = 'dotted', gap = 40):
    x, y, w, h = box
    tl = np.int16(np.array([x, y]))
    br = np.int16(np.array([x + w, y + h]))
    drawrect(img,tl,br,color,thickness = thickness, style=style, gap = gap)