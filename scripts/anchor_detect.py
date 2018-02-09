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
from utility import *

def detect_single(video_name, video_meta, face_list, com_list=None):
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
    
def detect_anchor(video_meta, single_person):
    fps = video_meta['fps']
    video_length = video_meta['video_length']
    
    FACE_SIM_THRESH = 1.1
    ## detect anchor by average similiarity 
    # sim = []
    # N = len(single_person)
    # for p1 in single_person:
    #     dist = 0
    #     for p2 in single_person:
    #         if p1 != p2:
    #             dist += np.linalg.norm(p1['feature'] - p2['feature'])
    #     sim.append(dist / (N-1))
    # top_id = np.argsort(sim)
    # anchor_person = []
    # for idx in top_id:
    #     if sim[idx] < FACE_SIM_THRESH:
    #         anchor_person.append(single_person[idx])
    #         print(single_person[idx]['fid'])
    #     else:
    #         break

    ## detect anchor by counting similiar faces 
    # sim = []
    # N = len(single_person)
    # for p1 in single_person:
    #     cnt = 0 
    #     for p2 in single_person:
    #         if p1 != p2:
    #             if np.linalg.norm(p1['feature'] - p2['feature']) < FACE_SIM_THRESH:
    #                 cnt += 1
    #     sim.append(cnt)

    # top_id = np.argsort(sim)[::-1]
    # for idx in top_id:
    #     print(single_person[idx]['fid'], sim[idx])

    # ## remove noise face by calculating average similiraty 
    # top_id = np.argmax(sim) #??? maybe most common is better
    # refer_face = single_person[top_id]
    # # print(refer_face['fid'])
    # anchor_person = []
    # for p in single_person:
    #     dist = np.linalg.norm(p['feature'] - refer_face['feature'])
    #     if dist < FACE_SIM_THRESH:
    #         anchor_person.append(p)
    # print(len(anchor_person))

    # anchor_person_raw = anchor_person
    # sim = []
    # for p1 in anchor_person_raw:
    #     dist = 0
    #     cnt = 0
    #     for p2 in anchor_person_raw:
    #         if p1 != p2:
    #             dist += np.linalg.norm(p1['feature'] - p2['feature'])
    #     sim.append(dist / (len(anchor_person_raw)-1))
    # top_id = np.argsort(sim)
    # anchor_person = []
    # for idx in top_id:
    # #     if sim[idx] < FACE_SIM_THRESH:
    #     anchor_person.append(anchor_person_raw[idx])
    #     print(anchor_person_raw[idx]['fid'], sim[idx])

    ## detect anchor by Kmeans
    NUM_CLUSTER = 5
    from sklearn.cluster import KMeans
    N = len(single_person)
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
    cluster = []
    for i in range(NUM_CLUSTER):
        group = []
        center = cluster_center[i]
        for p in single_person:
            if np.linalg.norm(p['feature'] - center['feature']) < FACE_SIM_THRESH:
                group.append(p)
        cluster.append(group)

    ## find the cluster with the most broad distribution
    BIN_WIDTH = 180
    SPREAD_THRESH = 0.5
    anchor_group = []
    for i in range(NUM_CLUSTER):
        bins = np.zeros(int(video_length / BIN_WIDTH)+1)
        for p in cluster[i]:
            t = get_second_from_fid(p['fid'], fps)
            bins[int(t / BIN_WIDTH)] += 1
        print("Cluster %d coverage = %f" % (i, 1. * np.count_nonzero(bins) / bins.shape[0]))    
        if 1. * np.count_nonzero(bins) / bins.shape[0] > SPREAD_THRESH:
            anchor_group.append(cluster[i])

    ## remove noise face by calculating average similiraty
    anchor_group_raw = anchor_group
    anchor_group = []
    for i, anchor_person_raw in enumerate(anchor_group_raw):
        sim = []
        for p1 in anchor_person_raw:
            dist = 0
            for p2 in anchor_person_raw:
                if p1 != p2:
                    dist += np.linalg.norm(p1['feature'] - p2['feature'])
            sim.append(dist / (len(anchor_person_raw)-1))
        top_id = np.argsort(sim)
        anchor_person = []
        ## remove noise at the end
        print("Anchor group %d: %d faces" % (i, len(anchor_person_raw)))
        for idx in top_id:
            anchor_person.append(anchor_person_raw[idx])
            print("fid: %d, similiarity = %f" % (anchor_person_raw[idx]['fid'], sim[idx]))
        anchor_group.append(anchor_person)

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
                    x1 = int(face['bbox']['bbox_x1'] * W)
                    y1 = int(face['bbox']['bbox_y1'] * H)
                    x2 = int(face['bbox']['bbox_x2'] * W)
                    y2 = int(face['bbox']['bbox_y2'] * H)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)
                    time = get_time_from_fid(fid, fps)
                    text = str(fid) + ':' + str(time)
                    cv2.rectangle(frame, (0, 0), (230, 30), color=(255,255,255), thickness=-1)
                    cv2.putText(frame, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,0,0), thickness=2)
        #             filename = '../tmp/'+video_name+'_'+'{:06d}'.format(fid)+'.jpg'
        #             cv2.imwrite(filename, frame)
                    anchor_all[i][j] = frame
                    break
        fid += 1
    cap.release()

    for i in range(len(anchor_all)):
        grid = view_grid(anchor_all[i], 5)
        H, W, C = grid.shape
        filename = '../tmp/anchor/' + video_name + str(i) + '.jpg'
        grid_small = cv2.resize(grid, (1080, int(H/W*1080)))
        cv2.imwrite(filename, grid_small)
    
def plot_anchor_distribution():
    pass