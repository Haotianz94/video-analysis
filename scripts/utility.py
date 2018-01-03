# help functions
def get_time_from_fid(fid, fps):
    second = int(fid / fps)
    minute = int(second / 60)
    hour = int(minute / 60)
    return (hour, minute % 60, second % 60)

def get_time_from_second(second):
    minute = int(second / 60)
    hour = int(minute / 60)
    return (hour, minute % 60, second % 60)

def get_second(time):
    return time[0]*3600 + time[1]*60 + time[2]

def get_time_difference(t1, t2):
    return (t2[0]*3600 + t2[1]*60 + t2[2]) - (t1[0]*3600 + t1[1]*60 + t1[2])

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
     