import numpy as np
import cv2
from skimage.util import view_as_blocks

def view_grid(im_in,ncols=3):
    n,h,w,c = im_in.shape
    dn = (-n)%ncols # trailing images
    im_out = (np.empty((n+dn)*h*w*c,im_in.dtype)
           .reshape(-1,w*ncols,c))
    view=view_as_blocks(im_out,(h,w,c))
    for k,im in enumerate( list(im_in) + dn*[0] ):
        view[k//ncols,k%ncols,0] = im 
    return im_out

def pad_image_height(img, bbox, size=None):
    H, W, C = img.shape
    newH = W * 2
    cy = int((bbox['bbox_y1'] + bbox['bbox_y2']) / 2)
    ratio = 1. * cy / newH - 1./3
    pad = int(1./3 * newH - cy)
    img_new = np.zeros((newH, W, 3))
    if pad == 0:
        img_new = img
    elif ratio < 0:
        y1 = pad
        y2 = min(newH, pad+H)
        img_new[y1 : y2] = img[:y2-y1]
    else:
        y1 = 0
        y2 = min(newH, H)
        img_new[y1 : y2] = img[:y2-y1]
    
    if not size is None:
        img_resize = resize_image(img_new, size, deform=True)
        return img_resize
    else:
        return img_new

def resize_image(img, size, deform=False):
    H, W, C = img.shape
    grid_h, grid_w = size
    if not deform:
        if 1. * H / W <= 2: 
            w = grid_w
            h = int(1. * H / W * w) 
        else:
            h = grid_h
            w = int(1. * W / H * h)
        img_resize = cv2.resize(img, (w, h))
    else:
        img_resize = cv2.resize(img, (grid_w, grid_h))
    return img_resize

def average_image(images, size=None):
    if len(images) == 0:
        return None
    if size is None:
        H, W, C = images[0].shape
    else:
        H, W = size
        C = images[0].shape[2]
    img_avg = np.zeros((H, W, C))
    for img in images:
        if not size is None:
            img_resize = resize_image(img, size, deform=True)
            img_avg += img_resize
        else:
            img_avg += img
    img_avg /= (1. * len(images))
    return img_avg

def stitch_img_grid(images, num_col, size=None, deform=False, average_row=False):
    if len(images) == 0:
        return None
    if not size is None:
        grid_h, grid_w = size
        C = images[0].shape[2]
    else:
        grid_h, grid_w, C = images[0].shape
        
    if average_row:
        num_row = len(images) / num_col
        images_np = np.zeros((len(images) + num_row, grid_h, grid_w, C))
    else:
        images_np = np.zeros((len(images), grid_h, grid_w, C))
    idx = 0
    images_row = []
    for i, im in enumerate(images):
        if im is None:
            continue
        if size is None:
            images_np[idx] = im
        else:
            img_resize = resize_image(im, (grid_h, grid_w), deform)
            h, w, c = img_resize.shape
            images_np[idx, :h, :w, :] = img_resize

        images_row.append(images_np[idx])
        idx += 1
        if average_row and i % num_col == num_col-1:
            images_np[idx] = average_img(images_row, None)
            idx += 1
            images_row = []
    if average_row:        
        grid = view_grid(images_np, num_col+1)
    else:
        grid = view_grid(images_np, num_col)
    return grid    

def get_detail_from_video_name(video_name):
    split = video_name.split('_')
    date = get_date_from_string(split[1], split[2])
    station = split[0][:-1]
    if station == 'CNN':
        show = video_name[21:]
    elif station == 'FOXNEWS':
        show = video_name[25:]
    elif station == 'MSNBC':
        show = video_name[23:]
    return date, station, show

def get_date_from_string(str1, str2):
    return (int(str1[:4]), int(str1[4:6]), int(str1[6:]), int(str2[:2]), int(str2[2:4]), int(str2[4:]))

def compare_date(date1, date2):
    if date1[0] < date2[0]:
        return -1
    elif date1[0] > date2[0]:
        return 1
    elif date1[1] < date2[1]:
        return -1
    elif date1[1] > date2[1]:
        return 1
    elif date1[2] < date2[2]:
        return -1
    elif date1[2] > date2[2]:
        return 1
    elif date1[2] == date2[2]:
        return 0

# help functions
def get_time_from_fid(fid, fps):
    second = int(fid / fps)
    minute = int(second / 60)
    hour = int(minute / 60)
    return (hour, minute % 60, second % 60)

def get_second_from_fid(fid, fps):
    second = int(fid / fps)
    return second

def get_fid_from_time(time, fps):
    second = time[0]*3600 + time[1]*60 + time[2]
    fid = int((second + 0.5) * fps)
    return fid

def get_time_from_second(second):
    minute = int(second / 60)
    hour = int(minute / 60)
    return (hour, minute % 60, second % 60)

def get_second(time):
    return time[0]*3600 + time[1]*60 + time[2]

def get_time_difference(t1, t2):
    return (t2[0]*3600 + t2[1]*60 + t2[2]) - (t1[0]*3600 + t1[1]*60 + t1[2])

def is_time_in_window(t, win):
    if get_time_difference(win[0], t) >= 0 and get_time_difference(t, win[1]) >= 0:
        return True
    else:
        return False

def calculate_overlap(a, b):
    t1 = get_time_difference(a[1], b[0])
    t2 = get_time_difference(b[1], a[0])
    if t1 >= 0 or t2 >= 0:
        return 0
    elif t1 < 0:
        t3 = get_time_difference(b[0], a[0])        
        t4 = get_time_difference(b[1], a[1])
        if t3 < 0:
            t3 = 0
        if t4 < 0:
            t4 = 0
        return -t1 - t3 - t4
    elif t2 < 0:
        t3 = get_time_difference(a[0], b[0])        
        t4 = get_time_difference(a[1], b[1])
        if t3 < 0:
            t3 = 0
        if t4 < 0:
            t4 = 0
        return -t2 - t3 - t4
    