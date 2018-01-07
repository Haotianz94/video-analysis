import numpy as np

# help functions
def get_time_from_fid(fid, fps):
    second = int(fid / fps)
    minute = int(second / 60)
    hour = int(minute / 60)
    return (hour, minute % 60, second % 60)

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

def insert_window(wa, wb):
    t1 = get_time_difference(wa[1][1], wb[0][1])
    t2 = get_time_difference(wb[1][1], wa[0][1])
    if t1 > 0 or t2 > 0:
        return False, wa
    else:
        t3 = get_time_difference(wa[0][1], wb[0][1])
        t4 = get_time_difference(wa[1][1], wb[1][1])
        if t3 >= 0:
            start = wa[0]
        else:
            start = wb[0]
        if t4 <= 0:
            end = wa[1]
        else:
            end = wb[1]
    return True, (start, end)
    
def insert_window2list(win_list, win):
    is_insert = False
    for i in range(len(win_list)):
        if get_time_difference(win_list[i][0][1], win[0][1]) < 0:
            win_list.insert(i, win)
            is_insert = True
            break
    if not is_insert:
        win_list.append(win)
    return win_list

def merge_commercial_list(list_a, list_b, avoid_long=False):
    MAX_COMMERCIAL_TIME = 250
    MAX_MERGE_GAP = 30
    # insert wb into wa
    merged_list = []
    insert_list = np.zeros(len(list_b))
    
    # remove potencial wb causing long merged block
    if avoid_long:
        i = 0
        while i < len(list_b):
            wb = list_b[i]
            delete = False
            for wa in list_a:
                if get_time_difference(wa[0][1], wb[1][1]) > MAX_COMMERCIAL_TIME and get_time_difference(wa[1][1], wb[0][1]) < MAX_MERGE_GAP:
                    del list_b[i]
                    delete = True
                    break
                if get_time_difference(wb[0][1], wa[1][1]) > MAX_COMMERCIAL_TIME and get_time_difference(wb[1][1], wa[0][1]) < MAX_MERGE_GAP:
                    del list_b[i]
                    delete = True
                    break
            if not delete:
                i += 1

    for wa in list_a:
        new_w = wa
        for i in range(len(list_b)):
#             old_w = new_w
            is_insert, new_w = insert_window(new_w, list_b[i])
#             if get_time_difference(new_w[0][1], new_w[1][1]) > MAX_COMMERCIAL_TIME:
#                 new_w = old_w
            if is_insert:
                insert_list[i] = 1
        merged_list.append(new_w)
    # add other single lowertext window
    for i in range(len(list_b)):
        if insert_list[i] == 0:
            insert_window2list(merged_list, list_b[i])
    
    #Todo: add more merge
    
    #merge small gaps
    i = 0
    while i < len(merged_list) - 1:
        if get_time_difference(merged_list[i][1][1], merged_list[i+1][0][1]) < 10:
            merged_list[i] = (merged_list[i][0], merged_list[i+1][1])
            del merged_list[i+1]
        else:
            i += 1
        
    return merged_list
    