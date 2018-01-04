import cv2
import numpy as np
import pysrt
import time
import pickle
from matplotlib import pyplot as plt
import re
from utility import *


# check functions
def check_black_in_gt(black_frame_list, groundtruth, fps):
    gt_pair = []
    for gt in groundtruth:
        gt_pair.append((get_second(gt[0]), get_second(gt[1])))
    for fid in black_frame_list:
#         print(fid)
        second = int(fid / fps)
        drop_in = False
        for gt in gt_pair:
            if second >= gt[0]-2 and second <= gt[1]+2:
                drop_in = True
#                 print(gt)
                break;
        if not drop_in:
            print(fid, get_time_from_fid(fid, fps))

# visualize functions
def visualize_commercial(commercial_list, groundtruth=None, raw_commercial_list=None, lowertext_window_list=None, blanktext_window_list=None):
    y_gt = [1, 1]
    y_com = [2, 2]
    y_raw = [3, 3]
    y_lower = [4, 4]
    y_blank = [5, 5]
    plt.figure(111)
    if not groundtruth is None: 
        for gt in groundtruth:
            text = 'gt:  ' + str(gt[0]) + '-' + str(gt[1])
            plt.plot([get_second(gt[0]), get_second(gt[1])], y_gt, 'g', linewidth=5.0, label=text)

    for com in commercial_list:
        text = 'our: ' + str(com[0][1]) + '-'+ str(com[1][1])
        plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_com, 'r', linewidth=5.0, label=text)
    
    if not raw_commercial_list is None: 
        for com in raw_commercial_list:
            text = 'our: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_raw, 'y', linewidth=5.0)

    if not lowertext_window_list is None:
        for com in lowertext_window_list:
            text = 'our: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_lower, 'b', linewidth=5.0)
    
    if not blanktext_window_list is None:
        for com in blanktext_window_list:
            text = 'our: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_blank, 'k', linewidth=5.0)
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([0, 10])
    plt.xlabel('video time (s)')
    cur_axes = plt.gca()
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()

def check_groundtruth(groundtruth, commercial_list):
    # calculate precision and recall
    sum_overlap = 0
    for gt in groundtruth:
        for com in commercial_list:
            sum_overlap += calculate_overlap(gt, (com[0][1], com[1][1]))
    sum_gt = 0
    for gt in groundtruth:
        sum_gt += get_time_difference(gt[0], gt[1])
    sum_com = 0
    for com in commercial_list:
        sum_com += get_time_difference(com[0][1], com[1][1])
    precision = 1.0 * sum_overlap / sum_com
    recall = 1.0 * sum_overlap / sum_gt
    print("Precision = %3f , Recall = %3f" %(precision*100, recall*100))
    print("===========================================================================================")
    return (precision, recall)

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
                commercial_end = (w[0], w[2])
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

# def filter_black_window(black_window_list, fps):
#     # filter black window by checking the beginning and end of the video (abandoned)
#     BEGINNING_SECOND_THRESH = 60
#     for w in black_window_list:
#         second = w[0] / fps
#         if second < BEGINNING_SECOND_THRESH:
#             black_window_list.remove(w)

def get_blanktext_window_list(transcript, video_desp):
    MIN_THRESH = 30
    MAX_THRESH = 240
    fps = video_desp['fps']
    blanktext_window_list = []
    for i in range(len(transcript)-1):
        time_span = get_time_difference(transcript[i][2], transcript[i+1][1])
        if time_span > MIN_THRESH and time_span < MAX_THRESH:
            blanktext_window_list.append(((get_fid_from_time(transcript[i][2],fps),transcript[i][2]),(get_fid_from_time(transcript[i+1][1],fps),transcript[i+1][1]) ))
    return blanktext_window_list
            
def is_lower_text(text):
    if len(text) == 0:
        return False
    lower = [c for c in text if c.islower()]
    if 1. * len(lower) / len(text) > 0.5:
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
                lower_end = (get_fid_from_time(t[2], fps), t[2])
        else:
            if not lower_end is None:
                lowertext_window_list.append((lower_start, lower_end))
                lower_start = None
                lower_end = None
    return lowertext_window_list

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

def remove_commercial_gaps(clist, transcript):
    i = 0
    while i < len(clist) - 1:
        if get_time_difference(clist[i][1][1], clist[i+1][0][1]) < 90:
            # add delay
            start_check_index = get_transcript_index_by_time(clist[i][1][1], transcript)
            end_check_index = get_transcript_index_by_time(clist[i+1][0][1], transcript) - 1
            text = ''
            for index in range(start_check_index, end_check_index+1):
                text += transcript[index][0]
            is_in_commercial = False    
            if text.find('>> Announcer:') != -1:
                is_in_commercial = True
            if is_in_commercial:    
                clist[i] = (clist[i][0], clist[i+1][1])
                del clist[i+1]
            else:
                i += 1
        else:
            i += 1
    return clist
    
def load_single_video(video_name):
    # load black_frame_dict from pickle
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", 'rb'))
    video_path = '../data/videos/' + video_name + '.mp4'
    srt_path = '../data/videos/' + video_name + '.cc5.srt'
    black_frame_list = black_frame_dict[video_name]

    # load transcript
    transcript = []
    subs = pysrt.open(srt_path)
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))

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
    
    return black_frame_list, transcript, video_desp
    
def get_black_frame_list(video_path):
    # load video
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(video_frames / fps)
    print(video_path, fps, video_length)

    fid = 0
    ret, frame = cap.read()
    pixel_sum = frame_w * frame_h
    BLACK_THRESH = 0.99 * pixel_sum
    black_frame_list = []

    start_time = time.time()
    while(ret):
        ret, frame = cap.read()
        if frame is None:
            break
        hist = cv2.calcHist([frame], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        #
    #         diff = get_time_difference(get_time_from_fid(fid, fps), (0,24,28))
    #         if np.fabs(diff) <= 1:
    #             cv2.imwrite('../tmp/frame_'+str(fid)+'.jpg', frame)
#         if fid == 44029:
#             print(hist)
        #
        if hist[0,0,0] > BLACK_THRESH:
            black_frame_list.append(fid)
        fid += 1
        if fid % 10000 == 0:
            print(fid)

    print("--- %s seconds ---" % (time.time() - start_time))        
    cap.release()
    return black_frame_list

# load commercial groundtruth
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

def test_single_video(video_name):
    commercial_gt = load_commercial_groundtruth()
    black_frame_list, transcript, video_desp = load_single_video(video_name)
    black_window_list = get_black_window_list(black_frame_list, video_desp)
    raw_commercial_list = search_commercial(black_window_list, transcript, video_desp)
    lowertext_window_list = get_lowertext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(raw_commercial_list, lowertext_window_list)
    blanktext_window_list = get_blanktext_window_list(transcript, video_desp)
    commercial_list = merge_commercial_list(commercial_list, blanktext_window_list)
    
    groundtruth = commercial_gt[video_name]['span']
    visualize_commercial(commercial_list, groundtruth, raw_commercial_list, lowertext_window_list)  
    res = check_groundtruth(groundtruth, commercial_list)
    
def test_video_list(video_list_path):
    commercial_gt = load_commercial_groundtruth()
    result = {}
    for line in open(video_list_path):
        video_name = line[:-1]
        print(video_name)
        black_frame_list, transcript, video_desp = load_single_video(video_name)
        black_window_list = get_black_window_list(black_frame_list, video_desp)
        raw_commercial_list = search_commercial(black_window_list, transcript, video_desp)
        lowertext_window_list = get_lowertext_window_list(transcript, video_desp)
        commercial_list = merge_commercial_list(raw_commercial_list, lowertext_window_list)
        blanktext_window_list = get_blanktext_window_list(transcript, video_desp)
        commercial_list = merge_commercial_list(commercial_list, blanktext_window_list)
        commercial_list = remove_commercial_gaps(commercial_list, transcript)

        groundtruth = commercial_gt[video_name]['span']
        visualize_commercial(commercial_list, groundtruth, raw_commercial_list, lowertext_window_list, blanktext_window_list)  
        res = check_groundtruth(groundtruth, commercial_list)
        result[video_name] = res
    return result

def prepare_black_frame_list(video_list_path):
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", "rb" ))
    for line in open(video_list_path):
        video_name = line[:-1]
        if not video_name in black_frame_dict:
            print(video_name)
            video_path = '../data/videos/' + video_name + '.mp4'
            srt_path = '../data/videos/' + video_name + '.cc5.srt'
            black_frame_list = get_black_frame_list(video_path)
            black_frame_dict[video_name] = black_frame_list
            pickle.dump(black_frame_dict, open("../data/black_frame_dict.p", "wb" ))