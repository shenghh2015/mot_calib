import numpy as np

def gen_kernel(ksize):
    base_kernel = np.array([1, 2, 1])
    conv_times = (ksize - 3) // 2
    kernel = base_kernel.copy()
    for i in range(conv_times):
        kernel = np.convolve(kernel, base_kernel, 'full')
    return kernel.flatten()

def smooth_1d(vec, ksize = 3, mode = 'same'):
    kernel = gen_kernel(ksize)
    kernel = np.array(kernel)
    kernel = kernel / np.sum(kernel)
    vec = np.convolve(vec, kernel, mode = mode)
    return vec

def smooth_boxes(boxes, ksize = 3):
    boxes = boxes.reshape((-1, 4))  # N x 4
    # top left, bottom, right
    tl_br = boxes.copy()
    tl_br[:, 2:] += tl_br[:,:2]
    for i in range(4):
        coord = tl_br[:, i]
        s_coord = smooth_1d(coord, ksize = ksize)
        tl_br[:, i] = s_coord
    tl_br[:,2:] -= tl_br[:,:2]
    return tl_br

def clip_boxes(boxes, clip_sizes = [3000, 20000]):
    box_sizes = boxes[:, :, 2] * boxes[:, :, 3]
    lower_size, upper_size = clip_sizes
    mask = np.expand_dims(np.logical_and(box_sizes < upper_size,\
            box_sizes > lower_size), axis = -1)
    return mask

def erosion_1d(ind_mask, ksize = 13):
    new_mask = np.zeros((len(ind_mask), 1))
    left, right, p = 0, 0, 0
    while p < len(ind_mask) and p < len(ind_mask):
        if ind_mask[p]:
            left = p
            while p < len(ind_mask) and ind_mask[p]: p += 1
            right = p
            # process mask
            if left + ksize - 1 < right - (ksize - 1):
                new_mask[left + ksize - 1: right - (ksize - 1)] = 1
            else:
                new_mask[left: right] = 0
        else:
            p += 1
    return new_mask

def mask_erosion(mask, ksize = 13):
    new_mask = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        new_mask[i, :, :] = erosion_1d(mask[i, :, 0], ksize)
    return new_mask

def smoothen_tracks(numpy_tracks, ksize = 13, use_clip = True):
    
    if use_clip:
        lower_size, upper_size = 1000, 200000
    else:
        lower_size, upper_size = 0, 1e8

    # obtain masks of tracks
    mask  = clip_boxes(numpy_tracks, clip_sizes = [lower_size, upper_size])
    mask  = mask_erosion(mask, ksize = ksize)

    # smooth tracks
    s_tracks = numpy_tracks.copy()
    for i in range(numpy_tracks.shape[0]):
        boxes = numpy_tracks[i, :, :]
        s_tracks[i, :, :] = smooth_boxes(boxes, ksize = ksize)  # smooth boxes for each track

    return s_tracks * mask

def smoothen_tracks_v2(numpy_tracks, ksize = 13, use_clip = True):
    
    if use_clip:
        lower_size, upper_size = 1000, 200000
    else:
        lower_size, upper_size = 0, 1e8

    # obtain masks of tracks
    mask  = clip_boxes(numpy_tracks, clip_sizes = [lower_size, upper_size])
    mask  = mask_erosion(mask, ksize = ksize)

    # smooth tracks
    s_tracks = numpy_tracks.copy()
    for i in range(numpy_tracks.shape[0]):
        boxes = numpy_tracks[i, :, :]
        s_tracks[i, :, :] = smooth_boxes(boxes, ksize = ksize)  # smooth boxes for each track

    mask_out = s_tracks * mask + numpy_tracks * (1 - mask)

    return mask_out

from utils import get_front_bboxes_nms
# all_bboxes: N x M x 4
def gen_front_mask(all_bboxes, h_img = 1080):
    output_mask = np.zeros(all_bboxes.shape[:2])
    for frame_id in range(all_bboxes.shape[1]):
        inds = get_front_bboxes_nms(all_bboxes[:, frame_id, :], h_img)
        output_mask[inds, frame_id] = 1.
    return output_mask