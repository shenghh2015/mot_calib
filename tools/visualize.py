import cv2
import numpy as np

def draw_text(img, text, pt, color = (255, 0, 0), scale = 10, thickness = 2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(pt[0]), int(pt[1])
    # Using cv2.putText() method
    img = cv2.putText(img, text, (x, y), font, 
                    scale, color, thickness, cv2.LINE_AA)
    return img

def draw_box(image, box, color, thickness = 2):
    x, y, w, h = box
    pt1 = np.int16(np.array([x, y]))
    pt2 = np.int16(np.array([x + w, y]))
    pt3 = np.int16(np.array([x + w, y + h]))
    pt4 = np.int16(np.array([x, y + h]))
    cv2.line(image, pt1, pt2, color, thickness)
    cv2.line(image, pt2, pt3, color, thickness)
    cv2.line(image, pt3, pt4, color, thickness)
    cv2.line(image, pt4, pt1, color, thickness)

def draw_boxes(image, boxes, color, thickness = 1):
    for i in range(len(boxes)):
        box = boxes[i,:].flatten()
        draw_box(image, box, color, thickness)

def create_blk_board(w_img, h_img, border = False):
    board = np.zeros((h_img, w_img, 3), dtype = np.uint8)
    if not border:
        return board
    else:
        return 