import cv2
import numpy as np
import pysrt
import time
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import math
import threading
import multiprocessing as mp
import codecs
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

def load_single_video_local(video_name):
    # load video meta data
    video_path = '../data/videos/' + video_name + '.mp4'
    video_file = Path(video_path)
    if not video_file.is_file():
        print("%s does not exist!!!" % video_path)
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(video_frames / fps)
    cap.release()
    video_desp = {'video_frames': video_frames, 'frame_w': frame_w, 'frame_h': frame_h, 'fps': fps, 'video_length': video_length}
    
    # load black_frame_dict from pickle
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", 'rb'))
    if not video_name in black_frame_dict:
        print("Black_frame_list does not exist!!!")
        return None, None, None
    black_frame_list = black_frame_dict[video_name]

    # load transcript
    srt_path = '../data/videos/' + video_name + '.cc5.srt'
    srt_file = Path(srt_path)
    if not srt_file.is_file():
        srt_path = srt_path.replace('cc5', 'cc1')
        srt_file = Path(srt_path)
        if not srt_file.is_file():
            print("%s does not exist!!!" % srt_path)
            return None, None, None
    
    # check uft-8
    try:
        file = codecs.open(srt_path, encoding='utf-8', errors='strict')
        for line in file:
            pass
    except UnicodeDecodeError:
        print("Transcript not encoded in utf-8!!!")
        return None, None, None
    
    transcript = []
    subs = pysrt.open(srt_path)
    text_length = 0
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))
        text_length += get_time_difference(transcript[-1][1], transcript[-1][2])            
    
    # check transcript completeness     
    if 1. * text_length / video_length < 0.3:
        print("Transcript not complete!!!")
        return None, None, None
        
    return black_frame_list, transcript, video_desp

def load_single_video(video_name, video_meta_dict, black_frame_dict):
    # load video meta data
    if not video_name in video_meta_dict:
        print("video_meta does not exist in dict!!!")
        return None, None, None
    video_desp = video_meta_dict[video_name]
    
    # load black_frame_dict from pickle
    if not video_name in black_frame_dict:
        print("black_frame_list does not exist in dict!!!")
        return None, None, None
    black_frame_list = black_frame_dict[video_name]

    # load transcript
#     srt_path = '../data/videos/' + video_name + '.cc5.srt'
    srt_path = '../data/transcripts/' + video_name + '.cc5.srt'

    srt_file = Path(srt_path)
    if not srt_file.is_file():
        srt_path = srt_path.replace('cc5', 'cc1')
        srt_file = Path(srt_path)
        if not srt_file.is_file():
            srt_path = srt_path.replace('cc1', 'align')
            srt_file = Path(srt_path)
            if not srt_file.is_file():
                print("%s does not exist!!!" % srt_path)
                return None, None, None
    
    # check uft-8
    try:
        file = codecs.open(srt_path, encoding='utf-8', errors='strict')
        for line in file:
            pass
    except UnicodeDecodeError:
        print("Transcript not encoded in utf-8!!!")
        return None, None, None
    
    transcript = []
    subs = pysrt.open(srt_path)
    text_length = 0
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))
        text_length += get_time_difference(transcript[-1][1], transcript[-1][2])            
    
    # check transcript completeness
    video_length = video_desp['video_length']
    if 1. * text_length / video_length < 0.3:
        print("Transcript not complete!!!")
        return None, None, None
        
    return black_frame_list, transcript, video_desp

def get_black_window_list(black_frame_list, video_desp):
#     print(black_frame_list)
    if len(black_frame_list) == 0:
        return []
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

def get_transcript_index_by_time(t, transcript, delay=0):
    # find the first index whose beginning time >= t
    for i in range(len(transcript)):
        stime = transcript[i][1]
        if get_time_difference(t, stime) - delay >= 0: 
            return i
    return len(transcript)-1   

def check_in_show(last_check_index, new_check_index, transcript):
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
    if len(black_window_list) == 0:
        return []
    # find commercial by checking arrows in transcript
    MIN_COMMERCIAL_TIME = 10
    MAX_COMMERCIAL_TIME = 270
    TRANSCRIPT_DELAY = 6
    
    video_length = video_desp['video_length']
    video_frames = video_desp['video_frames']
    fps = video_desp['fps']
    commercial_list = []
    commercial_start = (0, (0,0,0))
    last_check_index = 0
    commercial_end = None
    for w in black_window_list:
        new_check_index = get_transcript_index_by_time(w[2], transcript, 0) - 1
#             print(w[2])
        is_in_show = check_in_show(last_check_index, new_check_index, transcript)
        if is_in_show: 
            if commercial_end is None:
                pass
            else:
#                     print(commercial_start, commercial_end)
                if get_time_difference(commercial_start[1], commercial_end[1]) >= MIN_COMMERCIAL_TIME:
                    commercial_list.append((commercial_start, commercial_end))
            commercial_start = (w[0], w[2])
            commercial_end = None
            last_check_index = get_transcript_index_by_time(w[2], transcript, TRANSCRIPT_DELAY)
        else:
            if get_time_difference(commercial_start[1], w[2]) >= MAX_COMMERCIAL_TIME:
                if commercial_end != None:
                    pass
                else:
                    start_second = get_second(commercial_start[1])
                    end_second = start_second + MAX_COMMERCIAL_TIME
                    end_time = get_time_from_second(end_second)
                    commercial_end = (get_fid_from_time(end_time, fps), end_time)
                commercial_list.append((commercial_start, commercial_end))
                commercial_start = (w[0], w[2])
                commercial_end = None
                last_check_index = get_transcript_index_by_time(w[2], transcript, 0)
            else:
                commercial_end = (w[1], w[2])
                last_check_index = get_transcript_index_by_time(w[2], transcript, 0)
    if commercial_end != None:
        commercial_list.append((commercial_start, commercial_end))
    else:
        new_check_index = len(transcript) - 1
        is_in_show = check_in_show(last_check_index, new_check_index, transcript)
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
    MAX_THRESH = 270
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
    MIN_THRESH = 15
    MAX_GAP = 60
    for t in transcript:
        if is_lower_text(t[0]):
            if lower_start is None:
                lower_start = (get_fid_from_time(t[1], fps), t[1])
                lower_end = (get_fid_from_time(t[2], fps), t[2])
            else:
                if get_time_difference(lower_end[1], t[2]) < MAX_GAP: 
                    lower_end = (get_fid_from_time(t[2], fps), t[2])
                else:
                    if get_time_difference(lower_start[1], lower_end[1]) > MIN_THRESH:
                        lowertext_window_list.append((lower_start, lower_end))
                    lower_start = (get_fid_from_time(t[1], fps), t[1])
                    lower_end = (get_fid_from_time(t[2], fps), t[2])
        else:
            if not lower_end is None:
                if get_time_difference(lower_start[1], lower_end[1]) > MIN_THRESH:
                    lowertext_window_list.append((lower_start, lower_end))
                lower_start = None
                lower_end = None
    return lowertext_window_list

def post_process(clist, blist, transcript):    
    # remove_commercial_gaps
    GAP_THRESH = 90
    MAX_COMMERCIAL_TIME = 270
    i = 0
    while i < len(clist) - 1:
        if get_time_difference(clist[i][1][1], clist[i+1][0][1]) < GAP_THRESH and get_time_difference(clist[i][0][1], clist[i+1][1][1]) < MAX_COMMERCIAL_TIME:
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
    
    # remove small window and isolated blank window
    MIN_COMMERCIAL_TIME = 30
    MIN_BLANK_TIME = 90
    MAX_COMMERCIAL_TIME = 240
    i = 0
    while i < len(clist):
        is_delete = False
        span = get_time_difference(clist[i][0][1], clist[i][1][1])
        is_isolated = True
        if i-1 >= 0 and get_time_difference(clist[i-1][0][1], clist[i][1][1]) < MAX_COMMERCIAL_TIME:
            is_isolated = False
        if i+1 < len(clist) and get_time_difference(clist[i][0][1], clist[i+1][1][1]) < MAX_COMMERCIAL_TIME:
            is_isolated = False
        if span < MIN_COMMERCIAL_TIME:
            if is_isolated:
                del clist[i]
                is_delete = True
        elif span < MIN_BLANK_TIME:
            if clist[i] in blist and is_isolated:
                del clist[i]
                is_delete = True
        if not is_delete:
            i += 1
            
    return clist

def solve_single_video(video_name, video_meta_dict=None, black_frame_dict=None):
#     start_time = time.time()
    if video_meta_dict is None:
        black_frame_list, transcript, video_desp = load_single_video_local(video_name)
    else:
        black_frame_list, transcript, video_desp = load_single_video(video_name, video_meta_dict, black_frame_dict)
#     print("load time: --- %s seconds ---" % (time.time() - start_time))        
    
    if video_desp is None:
        return None, None, None, None, None
#     start_time = time.time()
    black_window_list = get_black_window_list(black_frame_list, video_desp)
    raw_commercial_list = search_commercial(black_window_list, transcript, video_desp)
    
    lowertext_window_list = get_lowertext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(raw_commercial_list, lowertext_window_list)
    blanktext_window_list = get_blanktext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(commercial_list, blanktext_window_list, avoid_long=True)
    commercial_list = post_process(commercial_list, blanktext_window_list, transcript)
    
#     print("compute time: --- %s seconds ---" % (time.time() - start_time))        
    return video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list

def count_arrows(clist, transcript):
    TRANSCRIPT_DELAY = 6
    i = 0
    num_arrows = []
    while i < len(clist) - 1:
        start_check_index = get_transcript_index_by_time(clist[i][1][1], transcript, 0)
        end_check_index = get_transcript_index_by_time(clist[i+1][0][1], transcript, TRANSCRIPT_DELAY) - 1
        text = ''
        for index in range(start_check_index, end_check_index+1):
            text += transcript[index][0]
        count = 0
        start_pos = -2
        while(True):
            start_pos = text.find('>>', start_pos + 2)
            if start_pos == -1:
                break
            else:
                count += 1
        num_arrows.append(count)
        i += 1
    return num_arrows

def test_single_video(video_name, local=True):
    commercial_gt = load_commercial_groundtruth()
    if local:
        video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name)
    else:
        video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
        black_frame_dict = pickle.load(open('../data/black_frame_dict.pkl', 'rb'))
        video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name, video_meta_dict, black_frame_dict)
    if video_desp is None:
        return 
    
    commercial_length = 0
    for com in commercial_list:
        commercial_length += get_time_difference(com[0][1], com[1][1])
    
    if video_name in commercial_gt:
        groundtruth = commercial_gt[video_name]['span']
        res = check_groundtruth(groundtruth, commercial_list)
        visualize_video_single(commercial_list, video_desp, groundtruth, raw_commercial_list, lowertext_window_list, blanktext_window_list)
        print("Precision = %3f , Recall = %3f, Commercial Length = %d" %(res[0]*100, res[1]*100, commercial_length)) 
    else:
        print("Commercial Length = %d" %(commercial_length)) 
        visualize_video_single(commercial_list, video_desp, None, raw_commercial_list, lowertext_window_list, blanktext_window_list)

def test_video_list(video_list_path, show_detail=False, local=True):
    commercial_gt = load_commercial_groundtruth()
    result = {}
    avg_precision = avg_recall = num_res = 0
    if not local:
        video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
        black_frame_dict = pickle.load(open('../data/black_frame_dict.pkl', 'rb'))
    for line in open(video_list_path):
        video_name = line[:-1]
#         print(video_name)
        if local:
            video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name)
        else:
            video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name, video_meta_dict, black_frame_dict)
        
        if video_desp is None:
            continue
            
        result[video_name] = {}
        result[video_name]['commercial'] = commercial_list
        
        commercial_length = 0
        for com in commercial_list:
            commercial_length += get_time_difference(com[0][1], com[1][1])
        result[video_name]['commercial_length'] = commercial_length
        
#         num_arrows = count_arrows(commercial_list, transcript)
#         result[video_name]['num_arrows'] = num_arrows
        
        if video_name in commercial_gt:
            groundtruth = commercial_gt[video_name]['span']
            res = check_groundtruth(groundtruth, commercial_list)
            result[video_name]['stat'] = res
            avg_precision += res[0]
            avg_recall += res[1]
            num_res += 1    
        
        if show_detail:
            if video_name in commercial_gt:
                groundtruth = commercial_gt[video_name]['span']
                visualize_video_single(commercial_list, video_desp, groundtruth, raw_commercial_list, lowertext_window_list, blanktext_window_list)
                print("Precision = %3f , Recall = %3f, Commercial Length = %d" %(res[0]*100, res[1]*100, commercial_length)) 
            else:
                visualize_video_single(commercial_list, video_desp, None, raw_commercial_list, lowertext_window_list, blanktext_window_list)
                print("Commercial Length = %d" %(commercial_length)) 
            print("===========================================================================================")
        
    if not show_detail:
        visualize_video_list(result, commercial_gt, 3700)
    if num_res > 0:
        print("Average precision = %3f , Average recall = %3f" %(avg_precision/num_res*100, avg_recall/num_res*100)) 

    return result

def test_all_result():
    commercial_gt = load_commercial_groundtruth()
    avg_precision = avg_recall = num_res = 0
    commercial_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    result = {}
    for video_name in sorted(commercial_dict):
        if video_name in commercial_gt:
            commercial_list = commercial_dict[video_name]
            result[video_name] = {}
            result[video_name]['commercial'] = commercial_list
            
            commercial_length = 0
            for com in commercial_list:
                commercial_length += get_time_difference(com[0][1], com[1][1])
            result[video_name]['commercial_length'] = commercial_length

            groundtruth = commercial_gt[video_name]['span']
            res = check_groundtruth(groundtruth, commercial_list)
            result[video_name]['stat'] = res
            avg_precision += res[0]
            avg_recall += res[1]
            num_res += 1    

    visualize_video_list(result, commercial_gt, 3700)
    if num_res > 0:
        print("Average precision = %3f , Average recall = %3f" %(avg_precision/num_res*100, avg_recall/num_res*100)) 

def search_commercial_t(video_list, com_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    commercial_dict = {}
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    black_frame_dict = pickle.load(open('../data/black_frame_dict.pkl', 'rb'))
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        video_desp, commercial_list, raw_commercial_list, lowertext_window_list, blanktext_window_list = solve_single_video(video_name, video_meta_dict, black_frame_dict)
        
        if video_desp is None:
            continue
            
        commercial_dict[video_name] = commercial_list
        if i % 100 == 0:
            pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
    pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))

def search_commercial_parallel(video_list_path, commercial_dict_path, nthread=16, use_process=True):
    video_list = open(video_list_path).read().split('\n')
    del video_list[-1]
    
    # remove exist videos:
    dict_file = Path(commercial_dict_path)
    if dict_file.is_file():
        commercial_dict = pickle.load(open(commercial_dict_path, "rb" ))
        i = 0
        while i < len(video_list):
            if video_list[i] in commercial_dict:
                del video_list[i]
            else:
                i += 1
    else:
        commercial_dict = {}
    
    num_video = len(video_list)
    print(num_video)
    if num_video <= nthread:
        nthread = num_video
        num_video_t = 1
    else:
        num_video_t = math.ceil(1. * num_video / nthread)
    print(num_video_t)
    
    commercial_dict_list = []
    for i in range(nthread):
        commercial_dict_list.append('../tmp/commercial_dict_' + str(i) + '.pkl')
    
    if use_process:
        ctx = mp.get_context('spawn')
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        if use_process:
            t = ctx.Process(target=search_commercial_t, args=(video_list_t, commercial_dict_list[i], i,))
        else:
            t = threading.Thread(target=search_commercial_t, args=(video_list_t, commercial_dict_list[i], i,))
            t.setDaemon(True)
        thread_list.append(t)
    
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    
    for path in commercial_dict_list:
        dict_file = Path(path)
        if not dict_file.is_file():
            continue
        commercial_dict_tmp = pickle.load(open(path, "rb" ))
        commercial_dict = {**commercial_dict, **commercial_dict_tmp}
        
    pickle.dump(commercial_dict, open(commercial_dict_path, "wb" ))
    
def post_process_result():
    commercial_dict = pickle.load(open('../data/commercial_dict_raw.pkl', 'rb'))
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    text_ratio_dict = pickle.load(open('../data/ratio_dict.pkl', 'rb'))
    
    total = 35722.0
    video_list = sorted(commercial_dict.keys())
    # remove video shorter than 10 min
    cnt = 0
    for video in video_list:
        if video in commercial_dict and video_meta_dict[video]['video_length'] < 600:
            cnt += 1
            del commercial_dict[video]
    print("short videos: ", cnt, cnt/total)
    
    # remove video missing transcript more than 40%
    cnt = 0
    for video in video_list:
        if video in commercial_dict and text_ratio_dict[video] < 0.6:
            cnt += 1
            del commercial_dict[video]
    print("incomplete transcript: ", cnt, cnt/total)
    
    # group all the videos by show name
    show_group_stat = {}
    for video_name in video_list:
        station_name = video_name.split('_')[0]
        if station_name == 'CNNW':
            show_name = video_name[21:]
        elif station_name == 'FOXNEWSW':
            show_name = video_name[25:]
        elif station_name == 'MSNBCW':
            show_name = video_name[23:]
        if not show_name in show_group_stat:
            show_group_stat[show_name] = []
        show_group_stat[show_name].append(video_name)
#         show_group_stat[show_name][video_name]['commercial_list'] = commercial_dict[video_name]

    # remove type of show which contains less than 10 videos
    cnt = 0
    for show in sorted(show_group_stat):
        num_videos = len(show_group_stat[show])
        if num_videos < 10:
            cnt += num_videos
            for video in show_group_stat[show]:
                if video in commercial_dict:
                    del commercial_dict[video]
    print("uncommon show: ", cnt, cnt/total)        
        
    # remove video which has commercial block more than 7 per hour
    cnt = 0
    for video in video_list:
        if video in commercial_dict:
            num_hour = np.round(1. * video_meta_dict[video]['video_length'] / 3600)
            if num_hour == 0:
                num_hour = 1
            coms = len(commercial_dict[video]) / num_hour
            if coms > 7:
                cnt += 1
                del commercial_dict[video]
    #             print(video, num_hour, coms)
    print("many commercial blocks: ", cnt, cnt/total)
    
    # remove video which has commercial coverage higher than 60%
    cnt = 0
    for video in video_list:
        if video in commercial_dict:
            com_len = 0
            for com in commercial_dict[video]:
                com_len += get_time_difference(com[0][1], com[1][1])
            cvg = 1. * com_len / video_meta_dict[video]['video_length']
            if cvg > 0.5:
                cnt += 1
    #             print(video_name, com_len, video_meta_dict[video_name]['video_length'])
                del commercial_dict[video]
    print("high commercial coverage: ", cnt, cnt/total)
    
    # remove Documentary
    cnt = 0
    for video in video_list:
        if video in commercial_dict and video.find('Dateline_Extra') != -1:
            cnt += 1
            del commercial_dict[video]
    print("documentary: ", cnt, cnt/total)
    
    # remove commercial blocks longer than video length
    cnt = 0
    for video in video_list:
        if video in commercial_dict:
            i = 0
            delete = False
            while i < len(commercial_dict[video]):
                com = commercial_dict[video][i]
                if get_second(com[1][1]) > video_meta_dict[video]['video_length']:
                    del commercial_dict[video][i]
                    delete = True
                else:
                    i += 1
            if delete:
                cnt += 1
    print("too long transcript: ", cnt)
    
    print(len(commercial_dict))
    pickle.dump(commercial_dict, open('../data/commercial_dict.pkl', 'wb'))