import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import time
from pathlib import Path
import codecs
import math
import multiprocessing as mp
from skimage.util import view_as_blocks
import copy
from sklearn.cluster import KMeans
import os
from utility import *

def detect_single(video_name, video_meta, face_list, com_list):
    fps = video_meta['fps']
    video_length = video_meta['video_length']
    # find shots with single face of an area above some threshhold
    MIN_FACE_AREA = 0
    single_person = []
    for shot in face_list:
        if len(shot['faces']) == 1:
            face = shot['faces'][0]
            bbox = face[0]
            if (bbox['bbox_x2'] - bbox['bbox_x1']) * (bbox['bbox_y2'] - bbox['bbox_y1']) > MIN_FACE_AREA: 
                single_person.append({'fid': shot['face_frame'], 'bbox': bbox, 'feature': np.array(face[1])})
#     print(len(single_person))

    # remove face in commercial
    if not com_list is None:
        single_person_raw = single_person
        single_person = []
        for p in single_person_raw:
            t = get_time_from_fid(p['fid'], fps)
            is_in_com = False
            for com in com_list:
                if is_time_in_window(t, (com[0][1], com[1][1])):
                    is_in_com = True
                    break
            if not is_in_com:
                single_person.append(p)
    #     print(len(single_person))
    return single_person
    
def detect_anchor(video_meta, single_person, com_list):
    fps = video_meta['fps']
    video_length = video_meta['video_length']
    
    FACE_SIM_THRESH = 1.0

    ## detect anchor by Kmeans
    NUM_CLUSTER = 5
    N = len(single_person)
    if N < NUM_CLUSTER:
        return []
    features = np.zeros((N, 128))
    for i in range(len(single_person)):
        features[i] = single_person[i]['feature']
    kmeans = KMeans(n_clusters=NUM_CLUSTER).fit(features)
    ## find the real cluster center in the dataset
    cluster_center = []
    for i in range(NUM_CLUSTER):
        c = kmeans.cluster_centers_[i]
        sim = []
        for p in single_person:
            dist = np.linalg.norm(p['feature'] - c)
            sim.append(dist)
        cluster_center.append(single_person[np.argmin(sim)])

    ## find same face for each cluster center
    FACE_SIM_SIDE_THRESH = 1.25
    cluster = []
    cluster_side = []
    for i in range(NUM_CLUSTER):
        group = []
        side = []
        center = cluster_center[i]
        for p in single_person:
            sim = np.linalg.norm(p['feature'] - center['feature'])
            if sim < FACE_SIM_THRESH:
#                 group.append(p)
                group.append(copy.deepcopy(p))
            elif sim < FACE_SIM_SIDE_THRESH:
#                 side.append(p)
                side.append(copy.deepcopy(p))
        cluster.append(group)
        cluster_side.append(side)
        
    ## remove noise face by calculating average similiraty
    for i in range(NUM_CLUSTER):
        sim = []
        for p1 in cluster[i]:
            dist = 0
            for p2 in cluster[i]:
                if p1['fid'] != p2['fid']:
                    dist += np.linalg.norm(p1['feature'] - p2['feature'])
            sim.append(dist / (len(cluster[i])-1))
        top_id = np.argsort(sim)
        new_group_idx = []
        ## remove noise at the end
        last_sim = sim[0] 
        for idx in top_id:
            if sim[idx] > 0.9 and sim[idx] - last_sim > 0.065:
#                 break
                ## collect side
                last_sim = 0
#                 s = np.linalg.norm(cluster_center[i]['feature'] - cluster[i][idx]['feature'])
#                 cluster[i][idx]['sim'] = s
                cluster_side[i].append(cluster[i][idx])
            else:
                new_group_idx.append(idx)
                last_sim = sim[idx]
#                 cluster[i][idx]['sim'] = sim[idx]
        new_group_idx.sort()
        new_group = []
        for idx in new_group_idx:
            new_group.append(cluster[i][idx])
        origin_faces = len(cluster[i])    
        cluster[i] = new_group
        print("Cluster %d: %d faces -> %d faces" % (i, origin_faces, len(cluster[i])))
    
    ## calculate the cluster coverage
    BIN_WIDTH = 300
    coverage = []
    anchor_group = []
    for i in range(NUM_CLUSTER):
        bins = np.zeros(int(video_length / BIN_WIDTH)+1)
        for p in cluster[i]:
            t = get_second_from_fid(p['fid'], fps)
            bins[int(t / BIN_WIDTH)] += 1
#         print("Cluster %d coverage = %f" % (i, 1. * np.count_nonzero(bins) / bins.shape[0]))    
        cov = 1. * np.count_nonzero(bins) / bins.shape[0]
        coverage.append(cov)
#         print("Cluster %d coverage: %f" % (i, coverage[i]))
    
    ## merge similar clusters
    MERGE_THRESH = 1.0
    while True:
        del_key = -1
        for i, c1 in enumerate(cluster_center):
            for j in range(i+1, len(cluster_center)):
                c2 = cluster_center[j]
                sim = np.linalg.norm(c1['feature']- c2['feature'])
                if sim < MERGE_THRESH:
                    print("Similarity between center %d and center %d = %f" % (i, j, sim))
                    if coverage[i] < coverage[j]:
                        del_key = i
                    else:
                        del_key = j
                    break
        if del_key != -1:
            del cluster[del_key]
            del cluster_center[del_key]
            del coverage[del_key]
            del cluster_side[del_key]
        else:
            break
    
    ## find the cluster with the most broad distribution
    SPREAD_THRESH = 0.45
    DURATION_THRESH = 0.55
    HEAD_APPEAR = 300
    anchor_group = []
    anchor_side = []
    anchor_center = []
    for i, c in enumerate(cluster):
        print("Cluster %d coverage: %f" % (i, coverage[i]))
        if coverage[i] < SPREAD_THRESH:
            continue
        duration = get_second_from_fid(c[-1]['fid'], fps) - get_second_from_fid(c[0]['fid'], fps)
        print("Cluster %d duration: %f" % (i, 1. * duration / video_length))
        if 1. * duration / video_length < DURATION_THRESH:
            continue
#         if com_list[0][0][0] < 
#         start = com_list[]
        anchor_group.append(c)
        anchor_side.append(cluster_side[i])
        anchor_center.append(cluster_center[i])
#     print("side", len(anchor_side[0]))
    
    ## reorder anchor_side
    for i, side in enumerate(anchor_side):
        for j, p in enumerate(anchor_side[i]):
            sim = np.linalg.norm(anchor_center[i]['feature'] - p['feature'])
            anchor_side[i][j]['sim'] = sim
        for j in range(len(side)):
            for k in range(j+1, len(side)):
                if side[j]['sim'] > side[k]['sim']:
                    tmp = anchor_side[i][j]
                    anchor_side[i][j] = anchor_side[i][k]
                    anchor_side[i][k] = tmp
            anchor_side[i][j]['fake'] = True
        for j, p in enumerate(anchor_group[i]):
            anchor_group[i][j]['fake'] = False
            sim = np.linalg.norm(anchor_center[i]['feature'] - p['feature'])
            anchor_group[i][j]['sim'] = sim
        anchor_group[i].extend(anchor_side[i])

    return anchor_group

def view_grid(im_in,ncols=3):
    n,h,w,c = im_in.shape
    dn = (-n)%ncols # trailing images
    im_out = (np.empty((n+dn)*h*w*c,im_in.dtype)
           .reshape(-1,w*ncols,c))
    view=view_as_blocks(im_out,(h,w,c))
    for k,im in enumerate( list(im_in) + dn*[0] ):
        view[k//ncols,k%ncols,0] = im 
    return im_out

def plot_anchor(video_name, video_meta, anchor_group):
    fps = video_meta['fps']
    video_path = '../data/videos/' + video_name + '.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    fid = 1
    H, W, C = frame.shape
    anchor_all = []
    for anchor_person in anchor_group:
        N = len(anchor_person)
        anchor_all.append(np.zeros((N, H, W, C)))
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(len(anchor_group)):
            anchor_person = anchor_group[i]
            for j in range(len(anchor_person)):
                face = anchor_person[j]
                if face['fid'] == fid:
                    img = copy.deepcopy(frame) 
                    x1 = int(face['bbox']['bbox_x1'] * W)
                    y1 = int(face['bbox']['bbox_y1'] * H)
                    x2 = int(face['bbox']['bbox_x2'] * W)
                    y2 = int(face['bbox']['bbox_y2'] * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)
                    time = get_time_from_fid(fid, fps)
                    text = str(fid) + '|' + str(time) + '|' + '{0:.3f}'.format(face['sim'])
                    cv2.rectangle(img, (0, 0), (320, 30), color=(255,255,255), thickness=-1)
                    cv2.putText(img, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,0,0), thickness=2)
                    if face['fake'] == True:
                        cv2.circle(img, (620, 20), 20, color=(255,255,255), thickness=-1)
        #             filename = '../tmp/'+video_name+'_'+'{:06d}'.format(fid)+'.jpg'
        #             cv2.imwrite(filename, frame)
                    anchor_all[i][j] = img
                    break
        fid += 1
    cap.release()

    for i in range(len(anchor_all)):
        grid = view_grid(anchor_all[i], 5)
        H, W, C = grid.shape
        filename = '../tmp/anchor/' + video_name + str(i) + '.jpg'
        grid_small = cv2.resize(grid, (1920, int(H/W*1920)))
        cv2.imwrite(filename, grid_small)
    
def plot_anchor_distribution(video_meta, anchor_group, com_list):
    fps = video_meta['fps']    
    video_length = video_meta['video_length']
    fig = plt.figure()
    fig.set_size_inches(14, 4)
    color = ['b', 'g', 'y', 'c', 'k']
    for i in range(len(anchor_group)):
        anchor_person = anchor_group[i]
        for p in anchor_person:
            if p['fake'] == False:
                t = get_second_from_fid(p['fid'], fps)
                plt.plot([t, t], [i+0.5, i+1.5], color[i], linewidth=1.0)
        for com in com_list:
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], [i+1, i+1], 'r', linewidth=4.0)

    plt.ylim([0, len(anchor_group)+1])
    plt.xlim([0, video_length])
    plt.xlabel('video time (s)')
    cur_axes = plt.gca()
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()
    
def solve_single_video(video_name, video_meta, face_list, com_list, plot=False):
    single_person = detect_single(video_name, video_meta, face_list, com_list)
#     for p in single_person:
#         print(get_time_from_fid(p['fid'], video_meta['fps']))
    anchor_group = detect_anchor(video_meta, single_person, com_list)
    if len(anchor_group) == 0:
        out = open('../log/detect_anchor.txt', 'a')
        out.write(video_name + '\n')
        out.close()
        return 
    if plot:
        plot_anchor(video_name, video_meta, anchor_group)
    else:
        plot_anchor_distribution(video_meta, anchor_group, com_list)
#     return anchor_group
#     print(np.linalg.norm(anchor_group[0][0]['feature'] - anchor_group[1][0]['feature']))    

def test_video_list(video_list_path):
    face_dict = pickle.load(open('../data/face_dict.pkl', 'rb'))
    com_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    for line in open(video_list_path):
        if len(line) < 5: 
            continue
        video_name = line[:-1]
        print(video_name)
        solve_single_video(video_name, meta_dict[video_name], face_dict[video_name], com_dict[video_name], True)

def test_single_video(video_name, plot=False):
    face_dict = pickle.load(open('../data/face_dict.pkl', 'rb'))
    com_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    solve_single_video(video_name, meta_dict[video_name], face_dict[video_name], com_dict[video_name], plot)
    
def detect_anchor_t(video_list, com_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
#     commercial_dict = {}
    face_dict = pickle.load(open('../data/face_dict.pkl', 'rb'))
    com_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        solve_single_video(video_name, meta_dict[video_name], face_dict[video_name], com_dict[video_name], True)
        
#         commercial_dict[video_name] = commercial_list
#         if i % 100 == 0:
#             pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
#     pickle.dump(commercial_dict, open(com_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))

def detect_anchor_parallel(video_list_path, commercial_dict_path=None, nthread=16, use_process=True):
    video_list = open(video_list_path).read().split('\n')
    
    # remove exist videos:
#     finished = []
#     for file in os.listdir('../tmp/anchor/'):
#         finished.append(file[:-5])
#     video_list = [video for video in video_list if video not in finished]

#     dict_file = Path(commercial_dict_path)
#     if dict_file.is_file():
#         commercial_dict = pickle.load(open(commercial_dict_path, "rb" ))
#         i = 0
#         while i < len(video_list):
#             if video_list[i] in commercial_dict:
#                 del video_list[i]
#             else:
#                 i += 1
#     else:
#         commercial_dict = {}
    
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
            t = ctx.Process(target=detect_anchor_t, args=(video_list_t, commercial_dict_list[i], i,))
        else:
            t = threading.Thread(target=detect_anchor_t, args=(video_list_t, commercial_dict_list[i], i,))
            t.setDaemon(True)
        thread_list.append(t)
    
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    
#     for path in commercial_dict_list:
#         dict_file = Path(path)
#         if not dict_file.is_file():
#             continue
#         commercial_dict_tmp = pickle.load(open(path, "rb" ))
#         commercial_dict = {**commercial_dict, **commercial_dict_tmp}
        
#     pickle.dump(commercial_dict, open(commercial_dict_path, "wb" ))    