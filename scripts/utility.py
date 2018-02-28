import numpy as np
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

def get_detail_from_video_name(video_name):
    split = video_name.split('_')
    date = get_date_from_string(split[1])
    station = split[0][:-1]
    if station == 'CNN':
        show = video_name[21:]
    elif station == 'FOXNEWS':
        show = video_name[25:]
    elif station == 'MSNBC':
        show = video_name[23:]
    return date, station, show

def get_date_from_string(str):
    return (int(str[:4]), int(str[4:6]), int(str[6:]))

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
    