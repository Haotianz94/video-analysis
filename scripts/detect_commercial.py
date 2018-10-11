import numpy as np
from pathlib import Path
import codecs
import pysrt

"""
All thresholds 
"""
TRANSCRIPT_DELAY = 6
MIN_TRANSCRIPT = 0.3

MIN_BLACKFRAME = 0.99
MIN_BLACKWINDOW = 1

MIN_BLANKWINDOW = 30
MAX_BLANKWINDOW = 270

MIN_LOWERTEXT = 0.5
MIN_LOWERWINDOW = 15
MAX_LOWERWINDOW_GAP = 60

MIN_COMMERCIAL_TIME = 10
MAX_COMMERCIAL_TIME = 270

MAX_MERGE_DURATION = 300
MAX_MERGE_GAP = 30
MIN_COMMERCIAL_GAP = 10
MAX_COMMERCIAL_GAP = 90

MIN_COMMERCIAL_TIME_FINAL = 30
MAX_ISOLATED_BLANK_TIME = 90


"""
Help functions for fid, time, second transfer
"""
def fid2second(fid, fps):
    second = 1. * fid / fps
    return second

def time2second(time):
    return time[0]*3600 + time[1]*60 + time[2]


"""
help functions with interval list operation 
"""
def merge_window(wa, wb):
    """
    merge wb with wa if they are connected
    """
    t1 = wb[0] - wa[1]
    t2 = wa[0] - wb[1]
    if t1 > 0 or t2 > 0:
        return False, wa
    else:
        t3 = wb[0] - wa[0]
        t4 = wb[1] - wa[1]
        if t3 >= 0:
            start = wa[0]
        else:
            start = wb[0]
        if t4 <= 0:
            end = wa[1]
        else:
            end = wb[1]
    return True, (start, end)
    
    
def insert_window2list(list_a, wb):
    """
    Insert wb into list_a (wb not connect with any wa in list_a)
    """
    is_inserted = False
    for i, wa in enumerate(list_a):
        if wb[0] - wa[0] < 0:
            list_a.insert(i, wb)
            is_inserted = True
            break
    if not is_inserted:
        list_a.append(wb)
    return list_a


def merge_list(list_a, list_b, avoid_long=False):
    """
    Insert wb from list_b into list_a
    """
    list_a = list_a.copy()
    list_b = list_b.copy()
    # Remove potencial wb causing long merged block
    list_b_filter = []
    if avoid_long:
        i = 0
        for wb in list_b:
            delete = False
            for wa in list_a:
                if wb[1] - wa[0] > MAX_MERGE_DURATION and wb[0] - wa[1] < MAX_MERGE_GAP:
                    delete = True
                    break
                if wa[1] - wb[0] > MAX_MERGE_DURATION and wa[0] - wb[1] < MAX_MERGE_GAP:
                    delete = True
                    break
            if not delete:
                list_b_filter.append(wb)
        list_b = list_b_filter
    
    merged_list = []
    is_inserted = np.zeros(len(list_b), dtype=bool)            
    # Merge list_b into list_a
    for wa in list_a:
        new_w = wa
        for i, wb in enumerate(list_b):
            if not is_inserted[i]:
                is_inserted[i], new_w = merge_window(new_w, wb)
        merged_list.append(new_w)
    # Add un-inserted window
    for i, wb in enumerate(list_b):
        if not is_inserted[i]:
            insert_window2list(merged_list, wb)
    
    # merge small gaps
    i = 0
    while i < len(merged_list) - 1:
        if merged_list[i+1][0] - merged_list[i][1] < MIN_COMMERCIAL_GAP:
            merged_list[i] = (merged_list[i][0], merged_list[i+1][1])
            del merged_list[i+1]
        else:
            i += 1
        
    return merged_list


def load_transcript(transcript_path, video_desp):
    """"
    Load trancript from *.srt file
    """
    # Check file exist
    filepath = Path(transcript_path)
    if not filepath.is_file():
        transcript_path = transcript_path.replace('cc5', 'cc1')
        filepath = Path(transcript_path)
        if not filepath.is_file():
            return None, "Transcript file does not exist!!!"
    
    # Check encoded in uft-8
    try:
        file = codecs.open(transcript_path, encoding='utf-8', errors='strict')
        for line in file:
            pass
    except UnicodeDecodeError:
        return None, "Transcript not encoded in utf-8!!!"
    
    transcript = []
    subs = pysrt.open(transcript_path)
    text_length = 0
    for sub in subs:
        transcript.append((sub.text, time2second(tuple(sub.start)[:3]), time2second(tuple(sub.end)[:3])))
        text_length += transcript[-1][2] - transcript[-1][1]            
    
    # Check transcript completeness     
    if 1. * text_length / video_desp['video_length'] < MIN_TRANSCRIPT:
        return None, "Transcript not complete!!!"
    
    return transcript, None


def get_blackframe_list(histogram, video_desp):
    """
    Get all black frames by checking the histogram list 
    """
    pixel_sum = video_desp['frame_w'] * video_desp['frame_h']
    thresh = MIN_BLACKFRAME * pixel_sum
    blackframe_list = []
    for fid, hist in enumerate(histogram):
        if hist[0][0] > thresh and hist[0][1] > thresh and hist[0][2] > thresh:
            blackframe_list.append(fid)
    return blackframe_list

        
def get_blackwindow_list(blackframe_list, video_desp):
    """
    Get black window by checking continuous black frames
    """
    if len(blackframe_list) == 0:
        return []
    fps = video_desp['fps']
    blackwindow_list = []
    fid = blackframe_list[0]
    nframe = 1
    for f in blackframe_list[1:]:
        if f == fid + nframe:
            nframe += 1
        else:
            if nframe >= MIN_BLACKWINDOW:
                black_window = (fid2second(fid, fps), fid2second(fid+nframe-1, fps))
                blackwindow_list.append(black_window)
            fid = f
            nframe = 1
    # Check black window at the end of video
    if nframe >= MIN_BLACKWINDOW:
        black_window = (fid2second(fid, fps), fid2second(fid+nframe-1, fps))
        blackwindow_list.append(black_window)
    
    return blackwindow_list


def get_transcript_index_by_time(t, transcript, delay=0):
    """
    Return the first index whose beginning time >= t
    """
    for i in range(len(transcript)):
        stime = transcript[i][1]
        if stime-delay - t  >= 0: 
            return i
    return len(transcript)-1   
    

def get_commercial_from_blackwindow(blackwindow_list, transcript, video_desp):
    """
    Find commercial block by checking << in transcript between black windows
    """
    def check_in_show(last_check_index, new_check_index, transcript):
        """
        Check if is in regular show by searching << between last_check_index and new_check_index
        """
        text = ''
        for i in range(last_check_index, new_check_index+1):
            text += transcript[i][0]
        start_pos = -2
        while(True):
            start_pos = text.find('>>', start_pos + 2)
            if start_pos == -1:
                is_in_show = False
                break
            else:
                arrow_announcer = text.find('>> Announcer:', start_pos, start_pos+15)
                if arrow_announcer == -1:
                    is_in_show = True
                    break
        return is_in_show

    if len(blackwindow_list) == 0:
        return []
    video_length = video_desp['video_length']
    video_frames = video_desp['video_frames']
    fps = video_desp['fps']
    commercial_list = []
    start, end = 0, None
    last_check_index = 0
    for w in blackwindow_list:
        new_check_index = get_transcript_index_by_time(w[0], transcript, 0) - 1
        is_in_show = check_in_show(last_check_index, new_check_index, transcript)
        if is_in_show: 
            if end is None:
                pass
            else:
                if end - start >= MIN_COMMERCIAL_TIME:
                    commercial_list.append((start, end))
            start = w[0]
            end = None
            last_check_index = get_transcript_index_by_time(w[1], transcript, TRANSCRIPT_DELAY)
        else:
            if w[1] - start >= MAX_COMMERCIAL_TIME:
                if end != None:
                    pass
                else:
                    end = start + MAX_COMMERCIAL_TIME
                commercial_list.append((start, end))
                start = w[0]
                end = None
                last_check_index = get_transcript_index_by_time(w[1], transcript, 0)
            else:
                end = w[1]
                last_check_index = get_transcript_index_by_time(w[1], transcript, 0)
    if end != None:
        commercial_list.append((start, end))
    else:
        new_check_index = len(transcript) - 1
        is_in_show = check_in_show(last_check_index, new_check_index, transcript)
        end = video_length
        if not is_in_show and end - start >= MIN_COMMERCIAL_TIME:
            commercial_list.append((start, end))
    return commercial_list


def get_commercial_from_blanktext(transcript, video_desp):
    """
    Get region without any transcript 
    """
    fps = video_desp['fps']
    blanktext_list = []
    for i in range(len(transcript)-1):
        start = transcript[i][2]
        end = transcript[i+1][1]
        time_span = end - start
        if time_span > MIN_BLANKWINDOW and time_span < MAX_BLANKWINDOW:
            end = end - TRANSCRIPT_DELAY
            blanktext_list.append((start, end))
    return blanktext_list
            

def get_commercial_from_lowertext(transcript, video_desp):
    """
    Get region with lower case transcript 
    """
    def is_lower_text(text):
        lower = [c for c in text if c.islower()]
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) == 0:
            return False
        if 1. * len(lower) / len(alpha) > MIN_LOWERTEXT:
            return True
        else:
            return False

    fps = video_desp['fps']
    lowertext_list = []
    start = end = None
    for t in transcript:
        if is_lower_text(t[0]):
            if start is None:
                start = t[1]
                end = t[2]
            else:
                if t[2] - end < MAX_LOWERWINDOW_GAP: 
                    end = t[2]
                else:
                    if end - start > MIN_LOWERWINDOW:
                        lowertext_list.append((start, end))
                    start = t[1]
                    end = t[2]
        else:
            if not end is None:
                if end - start > MIN_LOWERWINDOW:
                    lowertext_list.append((start, end))
                start = None
                end = None
    return lowertext_list


def post_process(clist, blist, transcript):
    """
    Final check merged commercial list
    """
    # Remove gaps between commercial blocks
    i = 0
    while i < len(clist) - 1:
        if clist[i+1][0] - clist[i][1] < MAX_COMMERCIAL_GAP and clist[i+1][1] - clist[i][0] < MAX_COMMERCIAL_TIME:
            # add delay
            start_check_index = get_transcript_index_by_time(clist[i][1], transcript)
            end_check_index = get_transcript_index_by_time(clist[i+1][0], transcript) - 1
            text = ''
            for index in range(start_check_index, end_check_index+1):
                text += transcript[index][0]
            is_in_commercial = True    
            if text.find('Announcer:') != -1:
                is_in_commercial = True
            elif text.find('>>') != -1:
                is_in_commercial = False
            if is_in_commercial:    
                clist[i] = (clist[i][0], clist[i+1][1])
                del clist[i+1]
                continue
        i += 1
    
    # Remove small window and isolated blank window
    i = 0
    while i < len(clist):
        delete = False
        span = clist[i][1] - clist[i][0]
        is_isolated = True
        if i-1 >= 0 and clist[i][1] - clist[i-1][0] < MAX_COMMERCIAL_TIME:
            is_isolated = False
        if i+1 < len(clist) and clist[i+1][1] - clist[i][0] < MAX_COMMERCIAL_TIME:
            is_isolated = False
        if span < MIN_COMMERCIAL_TIME_FINAL and is_isolated:
            del clist[i]
            delete = True
        elif span < MAX_ISOLATED_BLANK_TIME and clist[i] in blist and is_isolated:
            del clist[i]
            delete = True
        if not delete:
            i += 1
            
    return clist


def detect_commercial(video_desp, histogram, transcript):
    """
    API for detecting commercial blocks from TV news video
    Input:  video_desp (dict of fps, video_length, video_frames); 
            histogram (list of histogram 16x3 bin for each frame);  
            transcript (file path or a list of tuple(text, start_sec, end_sec));
    Return: commercial_list (list of tuple((start_fid, start_sec), (end_fid, end_sec)), None if failed)
            error_str (error message if failed, otherwise "")
    """
    if type(transcript) is str:
        transcript, error_str = load_transcript(transcript, video_desp)
    if transcript is None:
        return None, error_str
#     blackframe_list = get_blackframe_list(histogram)
    blackframe_list = histogram

    blackwindow_list = get_blackwindow_list(blackframe_list, video_desp)
    commercial_list = get_commercial_from_blackwindow(blackwindow_list, transcript, video_desp)

    lowertext_list = get_commercial_from_lowertext(transcript, video_desp)
    commercial_list = merge_list(commercial_list, lowertext_list)
    
    blanktext_list = get_commercial_from_blanktext(transcript, video_desp)
    commercial_list = merge_list(commercial_list, blanktext_list, avoid_long=True)
    commercial_list = post_process(commercial_list, blanktext_list, transcript)
    
    return commercial_list, ""