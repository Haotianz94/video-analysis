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
from commercial_search import *

def solve_single_video_coral(video_name, video_meta_dict, black_frame_dict, ground_truth=None):
    black_frame_list, transcript, video_desp = load_single_video(video_name, video_meta_dict, black_frame_dict)
    
    black_window_list = get_black_window_list(black_frame_list, video_desp)
    raw_commercial_list = search_commercial(black_window_list, transcript, video_desp)
    lowertext_window_list = get_lowertext_window_list(transcript, video_desp)
    blanktext_window_list = get_blanktext_window_list(transcript, video_desp)
    
    N = video_desp['video_length']
    print(N)
    if ground_truth is None:
        res = np.zeros((N, 3))
    else:
        res = np.zeros((N, 4))
    
    res[:, 0] = -1
    for com in raw_commercial_list:
        start = get_second(com[0][1])
        end = get_second(com[1][1])
        res[start:end, 0] = 1
    for com in lowertext_window_list:
        start = get_second(com[0][1])
        end = get_second(com[1][1])
        res[start:end, 1] = 1
    for com in blanktext_window_list:
        start = get_second(com[0][1])
        end = get_second(com[1][1])
        res[start:end, 2] = 1
    if not ground_truth is None:
        res[:, 3] = -1
        for gt in ground_truth:
            start = get_second(gt[0])
            end = get_second(gt[1])
            res[start:end, 3] = 1
    return res

def test_single_video_coral(video_name):
    commercial_gt = load_commercial_groundtruth()
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    black_frame_dict = pickle.load(open('../data/black_frame_dict.pkl', 'rb'))
    if video_name in commercial_gt:
        res = solve_single_video_coral(video_name, video_meta_dict, black_frame_dict, commercial_gt[video_name])
    else:
        res = solve_single_video_coral(video_name, video_meta_dict, black_frame_dict, None)
    return res

def prepare_commercial_coral_t(video_list, com_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    commercial_dict = {}
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    black_frame_dict = pickle.load(open('../data/black_frame_dict.pkl', 'rb'))
#     commercial_gt = load_commercial_groundtruth()
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        res = solve_single_video_coral(video_name, video_meta_dict, black_frame_dict, None)
        
        commercial_dict[video_name] = res
        if i % 100 == 0:
            pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
    pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))

def prepare_commercial_coral_parallel(video_list_path, commercial_dict_path, nthread=16, use_process=True):
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
    
#     commercial_gt = load_commercial_groundtruth()
#     video_gt = []
#     for video in video_list:
#         if video in commercial_gt:
#             video_gt.append(video)
#     video_list = video_gt
#     commercial_dict = {}
            
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
        commercial_dict_list.append('../tmp/commercial_dict_coral_' + str(i) + '.pkl')
    
    if use_process:
        ctx = mp.get_context('spawn')
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        if use_process:
            t = ctx.Process(target=prepare_commercial_coral_t, args=(video_list_t, commercial_dict_list[i], i,))
        else:
            t = threading.Thread(target=prepare_commercial_coral_t, args=(video_list_t, commercial_dict_list[i], i,))
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