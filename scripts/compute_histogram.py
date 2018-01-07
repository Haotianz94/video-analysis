import cv2
import numpy as np
import time
import pickle
import threading
from pathlib import Path
import math
from utility import *

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
        if hist[0,0,0] > BLACK_THRESH:
            black_frame_list.append(fid)
        fid += 1
        if fid % 10000 == 0:
            print(fid)

    print("--- %s seconds ---" % (time.time() - start_time))        
    cap.release()
    return black_frame_list

def prepare_black_frame_list(video_list_path):
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", "rb" ))
    for line in open(video_list_path):
        video_name = line[:-1]
        if len(video_name) < 5:
            continue
        if not video_name in black_frame_dict:
            print(video_name)
            video_path = '../data/videos/' + video_name + '.mp4'
            black_frame_list = get_black_frame_list(video_path)
            black_frame_dict[video_name] = black_frame_list
            pickle.dump(black_frame_dict, open("../data/black_frame_dict.p", "wb" ))

def get_black_frame_list_t(video_list, black_frame_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    black_frame_dict = {}
    i = 0
    for video_name in video_list:
        if len(video_name) < 5:
            continue
        print("Thread %d start %dth video..." % (thread_id, i))
        print(video_name)
        video_path = '../data/videos/' + video_name + '.mp4'
        black_frame_list = get_black_frame_list(video_path)
        black_frame_dict[video_name] = black_frame_list
        pickle.dump(black_frame_dict, open(black_frame_dict_path, "wb" ))
        i += 1
    print("Thread %d finished computing" % (thread_id))

        
def prepare_black_frame_list_multithread(video_list_path, nthread=4):
    video_list = open(video_list_path).read().split('\n')
    black_frame_dict = pickle.load(open("../data/black_frame_dict.p", "rb" ))
    # remove exist videos:
    for video in video_list:
        if video in black_frame_dict:
            video_list.remove(video)
    num_video = len(video_list)
    print(num_video)
    if num_video <= nthread:
        nthread = num_video
        num_video_t = 1
    else:
        num_video_t = math.ceil(1. * num_video / nthread)
    print(num_video_t)
    
    black_frame_dict_list = []
    for i in range(nthread):
        black_frame_dict_list.append('../tmp/black_frame_dict' + str(i) + '.p')
    
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        t = threading.Thread(target=get_black_frame_list_t, args=(video_list_t, black_frame_dict_list[i], i,))
        t.setDaemon(True)
        thread_list.append(t)
    
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    
    for path in black_frame_dict_list:
        dict_file = Path(path)
        if not dict_file.is_file():
            continue
        black_frame_dict_tmp = pickle.load(open(path, "rb" ))
        black_frame_dict = {**black_frame_dict, **black_frame_dict_tmp}
        
    pickle.dump(black_frame_dict, open("../data/black_frame_dict.p", "wb" ))