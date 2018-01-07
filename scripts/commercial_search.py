import cv2
import numpy as np
import pysrt
import time
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import math
from utility import *
from check import *
from compute_histogram import *

def load_commercial_groundtruth():
    commercial_gt = {}
    for line in open('../data/commercial_gt.csv'):
        columns = line[:-1].split(',')
        commercial_gt[columns[0]] = {'delay': float(columns[1]), 'span': []}
        for i in range(2, len(columns)):
            if columns[i] == '':
                continue
            span = columns[i].split('-')
            start = span[0].split(':')
            end = span[1].split(':')
            commercial_gt[columns[0]]['span'].append(((int(start[0]), int(start[1]), int(start[2])), (int(end[0]), int(end[1]), int(end[2]))))
    #print(commercial_gt)
    return commercial_gt

def load_single_video(video_name):
    # load black_frame_dict from pickle
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", 'rb'))
    
    video_path = '../data/videos/' + video_name + '.mp4'
    video_file = Path(video_path)
    if not video_file.is_file():
        print("%s does not exist!!!" % video_path)
        return None, None, None
    
    srt_path = '../data/videos/' + video_name + '.cc5.srt'
    srt_file = Path(srt_path)
    if not srt_file.is_file():
        srt_path = srt_path.replace('cc5', 'cc1')
        srt_file = Path(srt_path)
        if not srt_file.is_file():
            print("%s does not exist!!!" % srt_path)
            return None, None, None
    
    if not video_name in black_frame_dict:
        print("black_frame_list does not exist!!!")
        return None, None, None
    
    black_frame_list = black_frame_dict[video_name]
    # load transcript
    transcript = []
    subs = pysrt.open(srt_path)
    text_length = 0
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))
        text_length += get_time_difference(transcript[-1][1], transcript[-1][2])            
    
    # load video
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(video_frames / fps)
    cap.release()
    video_desp = {'video_frames': video_frames, 'frame_w': frame_w, 'frame_h': frame_h, 'fps': fps, 'video_length': video_length}

    # check black_frame_list in groundtruth
#     groundtruth = commercial_gt[video_name]['span']
#     check_black_in_gt(black_frame_list, groundtruth, fps)
    
    # check completeness     
    if 1. * text_length / video_length < 0.3:
        print("Transcript not complete!!!")
        return None, None, None
    
    return black_frame_list, transcript, video_desp

def get_black_window_list(black_frame_list, video_desp):
    # get black window by checking continuous black frames numbers
    CONTINUOUS_THRESH = 1
    fps = video_desp['fps']
    black_window_list = []
    fid = -99
    nframe = 1
    for f in black_frame_list:
        if f == fid + nframe:
            nframe += 1
        else:
            if nframe >= CONTINUOUS_THRESH and fid != -99:
                black_time = get_time_from_fid((fid*2+nframe-1)/2, fps)
                black_window = (fid, fid+nframe-1, black_time)
                black_window_list.append(black_window)
            fid = f
            nframe = 1
    if nframe >= CONTINUOUS_THRESH:
        black_time = get_time_from_fid((fid*2+nframe-1)/2, fps)
        black_window = (fid, fid+nframe-1, black_time)
        black_window_list.append(black_window)
    
#     for bw in black_window_list:
#         print(bw)
    return black_window_list

def get_transcript_index_by_time(t, transcript):
    # find the first index whose beginning time >= t
    TRANSCRIPT_DELAY = 3
    for i in range(len(transcript)):
        stime = transcript[i][1]
        if get_time_difference(t, stime) - TRANSCRIPT_DELAY >= 0: 
            return i
    return len(transcript)-1   

def find_arrow_in_show(last_check_index, new_check_index, transcript):
    text = ''
    for i in range(last_check_index, new_check_index+1):
        text += transcript[i][0]
#     print(text)
    #Todo: add 'Bill: '
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
#     print(is_in_show)
    return is_in_show

def search_commercial(black_window_list, transcript, video_desp):
    # find commercial by checking arrows in transcript
    MIN_COMMERCIAL_TIME = 10
    MAX_COMMERCIAL_TIME = 240
    
    video_length = video_desp['video_length']
    video_frames = video_desp['video_frames']
    commercial_list = []
    commercial_start = (0, (0,0,0))
    last_check_index = 0
    commercial_end = None
    for w in black_window_list:
        if commercial_start is None:
            commercial_start = (w[0], w[2]) # count start from the beginning of the black window
            last_check_index = get_transcript_index_by_time(w[2], transcript)
        else:
            new_check_index = get_transcript_index_by_time(w[2], transcript) - 1
#             print(w[2])
            is_in_show = find_arrow_in_show(last_check_index, new_check_index, transcript)
            if is_in_show: 
                if commercial_end is None:
                    pass
                else:
#                     print(commercial_start, commercial_end)
                    if get_time_difference(commercial_start[1], commercial_end[1]) >= MIN_COMMERCIAL_TIME:
                        commercial_list.append((commercial_start, commercial_end))
                commercial_start = (w[0], w[2])
                commercial_end = None
                last_check_index = new_check_index + 1
            else:
                if get_time_difference(commercial_start[1], w[2]) >= MAX_COMMERCIAL_TIME:
                    if commercial_end != None:
                        pass
                    else:
                        start_second = get_second(commercial_start[1])
                        end_second = start_second + MAX_COMMERCIAL_TIME
                        end_time = get_time_from_second(end_second)
                        commercial_end = (get_fid_from_time(end_time), end_time)
                    commercial_list.append((commercial_start, commercial_end))
                    commercial_start = (w[0], w[2])
                    commercial_end = None
                    last_check_index = new_check_index + 1
                else:
                    commercial_end = (w[1], w[2])
                    last_check_index = new_check_index + 1
    if commercial_end != None:
        commercial_list.append((commercial_start, commercial_end))
    elif commercial_start != None:
        new_check_index = len(transcript) - 1
        is_in_show = find_arrow_in_show(last_check_index, new_check_index, transcript)
        commercial_end = (video_frames-1, get_time_from_second(video_length))
        if not is_in_show and get_time_difference(commercial_start[1], commercial_end[1]) >= MIN_COMMERCIAL_TIME:
#             print(commercial_start, commercial_end)
            commercial_list.append((commercial_start, commercial_end))

#     for commercial in commercial_list:
#         print(commercial)
    return commercial_list

def get_blanktext_window_list(transcript, video_desp):
    TRANSCRIPT_DELAY = 3
    MIN_THRESH = 30
    MAX_THRESH = 240
    fps = video_desp['fps']
    blanktext_window_list = []
    for i in range(len(transcript)-1):
        start = transcript[i][2]
        end = transcript[i+1][1]
        time_span = get_time_difference(start, end)
        if time_span > MIN_THRESH and time_span < MAX_THRESH:
            second = get_second(end) - TRANSCRIPT_DELAY
            end = get_time_from_second(second)
            blanktext_window_list.append(((get_fid_from_time(start,fps), start),(get_fid_from_time(end,fps),end) ))
    return blanktext_window_list
            
def is_lower_text(text):
    lower = [c for c in text if c.islower()]
    alpha = [c for c in text if c.isalpha()]
    if len(alpha) == 0:
        return False
    if 1. * len(lower) / len(alpha) > 0.5:
        return True
    else:
        return False

def get_lowertext_window_list(transcript, video_desp):
    fps = video_desp['fps']
    lowertext_window_list = []
    lower_start = lower_end = None
    for t in transcript:
        if is_lower_text(t[0]):
            if lower_start is None:
                lower_start = (get_fid_from_time(t[1], fps), t[1])
                lower_end = (get_fid_from_time(t[2], fps), t[2])
            else:
                #Todo: add thresh 
                lower_end = (get_fid_from_time(t[2], fps), t[2])
        else:
            if not lower_end is None:
                lowertext_window_list.append((lower_start, lower_end))
                lower_start = None
                lower_end = None
    return lowertext_window_list

def post_process(clist, transcript):
    MIN_COMMERCIAL_TIME = 10
    # remove small window
    i = 0
    while i < len(clist):
        if get_time_difference(clist[i][0][1], clist[i][1][1]) < MIN_COMMERCIAL_TIME:
            del clist[i]
        else:
            i += 1
    
    # remove_commercial_gaps
    GAP_THRESH = 90
    i = 0
    while i < len(clist) - 1:
        if get_time_difference(clist[i][1][1], clist[i+1][0][1]) < GAP_THRESH:
            # add delay
            start_check_index = get_transcript_index_by_time(clist[i][1][1], transcript)
            end_check_index = get_transcript_index_by_time(clist[i+1][0][1], transcript) - 1
            text = ''
            for index in range(start_check_index, end_check_index+1):
                text += transcript[index][0]
#             print(clist[i][1][1], text)
            is_in_commercial = True    
            if text.find('Announcer:') != -1:
                is_in_commercial = True
            elif text.find('>>') != -1:
                is_in_commercial = False
            if is_in_commercial:    
                clist[i] = (clist[i][0], clist[i+1][1])
                del clist[i+1]
            else:
                i += 1
        else:
            i += 1
    return clist

def solve_single_video(video_name):
    black_frame_list, transcript, video_desp = load_single_video(video_name)
    if video_desp is None:
        return None, None, None, None, None
    black_window_list = get_black_window_list(black_frame_list, video_desp)
    raw_commercial_list = search_commercial(black_window_list, transcript, video_desp)
#     return video_desp, raw_commercial_list, None, None, None
    
    lowertext_window_list = get_lowertext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(raw_commercial_list, lowertext_window_list)
    blanktext_window_list = get_blanktext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(commercial_list, blanktext_window_list)
    commercial_list = post_process(commercial_list, transcript)
    return video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list

def test_single_video(video_name):
    commercial_gt = load_commercial_groundtruth()
    video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name)
    if video_desp is None:
        return 
    
    if video_name in commercial_gt:
        groundtruth = commercial_gt[video_name]['span']
        res = check_groundtruth(groundtruth, commercial_list)
        result[video_name]['stat'] = res
        visualize_video_single(commercial_list, video_desp, groundtruth, raw_commercial_list, lowertext_window_list, blanktext_window_list)  
    else:
        visualize_video_single(commercial_list, video_desp, None, raw_commercial_list, lowertext_window_list, blanktext_window_list)

        
def test_video_list(video_list_path):
    commercial_gt = load_commercial_groundtruth()
    result = {}
    for line in open(video_list_path):
        video_name = line[:-1]
        print(video_name)
        result[video_name] = {}
        video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name)
        if video_desp is None:
            continue
            
        result[video_name]['commercial'] = commercial_list
        if video_name in commercial_gt:
            groundtruth = commercial_gt[video_name]['span']
            res = check_groundtruth(groundtruth, commercial_list)
            result[video_name]['stat'] = res
            visualize_video_single(commercial_list, video_desp, groundtruth, raw_commercial_list, lowertext_window_list, blanktext_window_list)  
        else:
            visualize_video_single(commercial_list, video_desp, None, raw_commercial_list, lowertext_window_list, blanktext_window_list)
        
        print("===========================================================================================")
    return result


def test_video_list_unsupervised(video_list_path):
    commercial_gt = load_commercial_groundtruth()
    result = {}
    for line in open(video_list_path):
        video_name = line[:-1]
#         print(video_name)
        video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name)
        if video_desp is None:
            continue
        
        result[video_name] = {}
        result[video_name]['commercial'] = commercial_list
#         visualize_video_single(commercial_list, video_desp, None, raw_commercial_list, lowertext_window_list, blanktext_window_list)
        commercial_length = 0
        for com in commercial_list:
            commercial_length += get_time_difference(com[0][1], com[1][1])
        result[video_name]['commercial_length'] = commercial_length
    
#         commercial_ratio = 1. * commercial_length / video_desp['video_length'] 
#         print("commercial length =  %d  commercial ratio = %3f" % (commercial_length, commercial_ratio*100))
#         print("===========================================================================================")
    
#     visualize_video_list(result, video_desp)
    
    return result, video_desp
    