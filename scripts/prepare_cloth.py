import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import time
from pathlib import Path
import codecs
import math
import multiprocessing as mp
# from skimage.util import view_as_blocks
import copy
# from sklearn.cluster import KMeans
import os
from hwang import Decoder
from storehouse import StorageConfig, StorageBackend, RandomReadFile
from utility import *

def prepare_cloth_histogram(video_name, anchor_group):

    ## collect face by fid
    fid2person = {}
    dist_matrix = []
    num_face = 0
    for i, anchor_person in enumerate(anchor_group):
        for j, p in enumerate(anchor_person):
            if not p['fake']:
                if not p['fid'] in fid2person:
                    fid2person[p['fid']] = []
                p['index'] = (i, j)
                fid2person[p['fid']].append(p)
                num_face += 1
        dist_matrix.append([])

    video_path = '../data/videos/' + video_name + '.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Video %s does not exist!!!" % video_path)
        return 
    fid = 1

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = 0
    while cnt != num_face and ret:
        ret, frame = cap.read()
        if fid in fid2person:
            for p in fid2person[fid]:
                index = p['index']
                cnt += 1
                x1 = int(p['bbox']['bbox_x1'] * W)
                y1 = int(p['bbox']['bbox_y1'] * H)
                x2 = int(p['bbox']['bbox_x2'] * W)
                y2 = int(p['bbox']['bbox_y2'] * H)
                ## set crop window
                crop_w = (x2 - x1) * 2
                crop_h = crop_w * 2
                X1 = int((x1 + x2) / 2 - crop_w / 2)
                X2 = X1 + crop_w
                Y1 = int((y1 + y2) / 2 - crop_h / 3)
                Y2 = Y1 + crop_h
                ## adjust box size by image boundary
                crop_x1 = max(0, X1)
                crop_x2 = min(W-1, X2)
                crop_y1 = max(0, Y1)
                crop_y2 = min(H-1, Y2)
                ## compare histogram to find obstacle
                y_med = int((y2 + crop_y2) / 2)
                hist_top = cv2.calcHist([frame[y2:y_med, crop_x1:crop_x2+1, :]], [0, 1, 2], None, [16, 16, 16], [0, 255, 0, 255, 0, 255])
                hist_bottom = cv2.calcHist([frame[y_med:crop_y2+1, crop_x1:crop_x2+1, :]], [0, 1, 2], None, [16, 16, 16], [0, 255, 0, 255, 0, 255])
                ## normalization
                hist_top /= np.linalg.norm(hist_top)
                hist_bottom /= np.linalg.norm(hist_bottom)
                
                dist = np.linalg.norm(hist_top - hist_bottom)
                dist_matrix[index[0]].append(dist)
                cropped = frame[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]
#                 text = '{0:.3f}'.format(dist)
#                 cv2.rectangle(cropped, (0, 0), (cropped.shape[1]-1, 30), color=(255,255,255), thickness=-1)
#                 cv2.putText(cropped, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,0,0), thickness=2)
                p['frame'] = cropped
        fid += 1
    cap.release()

    res = []
    NUM_ANCHOR = 5
    folder = '../data/cloth_test'
    os.mkdir(folder)
    for i, anchor_person in enumerate(anchor_group):
        top_sim = np.argsort(dist_matrix[i])
        num_anchor = min(NUM_ANCHOR, len(anchor_person))
        res.append([])
        for j in range(num_anchor):
            filename = video_name + '_' + str(i) + '_' + str(j) + '.jpg'
            cv2.imwrite(os.path.join(folder, filename), anchor_person[top_sim[j]]['frame'])
            res[-1].append(filename)
    return res

def detect_text(img, start_y=40):
    img_bright = np.max(img, axis=2)
    H, W = img_bright.shape
    img_bright = img_bright.astype('int')
    img_contrast = np.zeros(img_bright.shape)
    CONTRAST_THRESH = 96
    TEXT_THRESH = 0.45
    HEAD_THRESH = 0.3
    for y in range(start_y, H):
        cnt_grad = 0
        for x in range(W):
            grad_horiz = False
            grad_verti = False
            neighbor = [-2, -1, 1, 2]
            for i in neighbor:
                if x+i < 0 or x+i > W-1:
                    continue
                if np.fabs(img_bright[y, x+i] - img_bright[y, x]) > CONTRAST_THRESH:
                    grad_horiz = True
                    break
#             for i in neighbor:        
#                 if y+i < 0 or y+i > H-1:
#                     continue
#                 if np.fabs(img_bright[y+i, x] - img_bright[y, x]) > CONTRAST_THRESH:
#                     grad_verti = True
#                     break
            if not grad_horiz: # or not grad_verti:
                img_contrast[y, x] = 200
            else:
                cnt_grad += 1
        if 1. * cnt_grad / W > TEXT_THRESH:
            ## too close to head: possibly wearing texture
#             if 1. * (y-start_y) / (H-start_y) < HEAD_THRESH:
#                 return img_contrast
            img_contrast[y, :] = 0
#             break
    return img_contrast

def detect_boundary(img, start_y=1):
    img_bright = np.max(img, axis=2)
    img_out = copy.deepcopy(img)
    H, W = img_bright.shape
    BOUNDARY_THRESH = 215
    HEAD_THRESH = 0.3
    dist_list = []
    for y in range(start_y, H):
        dist = np.linalg.norm(img_bright[y-2] - img_bright[y]) / np.sqrt(W)
        dist_list.append(dist)
        if dist > BOUNDARY_THRESH:
            ## too close to head: possibly wearing texture
#             if 1. * (y-start_y) / (H-start_y) < HEAD_THRESH:
#                 return img_out
            img_out[y-2:y+1, :, :] = [0, 255, 255]
    
    sort_dist = np.argsort(dist_list)[::-1]
    for i in range(30):
        print(dist_list[sort_dist[i]])
    print('+++++++++++++++++++++++++++++++')
            
    return img_out

def detect_edge(img, start_y=1):
    img_out = copy.deepcopy(img)
    edges = cv2.Canny(img, 80, 80)
    H, W = edges.shape
    BOUNDARY_THRESH = 0.5
    HEAD_THRESH = 0.3
    for y in range(start_y, H):
        non_zero = np.count_nonzero(edges[y])
        if 1. * non_zero / W > BOUNDARY_THRESH:
            ## too close to head: possibly wearing texture
#             if 1. * (y-start_y) / (H-start_y) < HEAD_THRESH:
#                 return img_out
            img_out[y:y+3, :, :] = [0, 255, 255]
    return edges

def detect_edge_text(img, start_y=40):
    edges = cv2.Canny(img, 80, 80)
    img_bright = np.max(img, axis=2)
    H, W = img_bright.shape
    img_bright = img_bright.astype('int')
#     img_contrast = np.zeros(img_bright.shape)
    BOUNDARY_THRESH = 0.5
    CONTRAST_THRESH = 96
    TEXT_THRESH = 0.45
    HEAD_THRESH = 0.3
    start_y = int((H - start_y) * HEAD_THRESH + start_y)
    for y in range(start_y, H):
        ## detect edge
        find_edge = False
        find_text = False
        non_zero = np.count_nonzero(edges[y])
        if 1. * non_zero / W > BOUNDARY_THRESH:
            find_edge = True
        ## find text    
        cnt_grad = 0
        for x in range(W):
            grad_horiz = False
            neighbor = [-2, -1, 1, 2]
            for i in neighbor:
                if x+i < 0 or x+i > W-1:
                    continue
                if np.fabs(img_bright[y, x+i] - img_bright[y, x]) > CONTRAST_THRESH:
                    grad_horiz = True
                    break
            if grad_horiz:
                cnt_grad += 1
        if 1. * cnt_grad / W > TEXT_THRESH:
            find_text = True
        if find_edge or find_text:    
            ## too close to head: possibly wearing texture
#             if 1. * (y-start_y) / (H-start_y) < HEAD_THRESH:
#                 return -1
#             img_out[y:y+3, :, :] = [0, 255, 255]
            return y
    return H

def prepare_cloth_crop_local(video_name, anchor_group):

    ## collect face by fid
    fid2person = {}
    ratio_matrix = []
    num_face = 0
    for i, anchor_person in enumerate(anchor_group):
        for j, p in enumerate(anchor_person):
            if not p['fake']:
                if not p['fid'] in fid2person:
                    fid2person[p['fid']] = []
                p['index'] = (i, j)
                fid2person[p['fid']].append(p)
                num_face += 1
        ratio_matrix.append([])

    video_path = '../data/videos/' + video_name + '.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Video %s does not exist!!!" % video_path)
        return 
    fid = 1

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = 0
    while cnt != num_face and ret:
        ret, frame = cap.read()
        if fid in fid2person:
            for p in fid2person[fid]:
                index = p['index']
                cnt += 1
                x1 = int(p['bbox']['bbox_x1'] * W)
                y1 = int(p['bbox']['bbox_y1'] * H)
                x2 = int(p['bbox']['bbox_x2'] * W)
                y2 = int(p['bbox']['bbox_y2'] * H)
                ## set crop window
                crop_w = (x2 - x1) * 2
                crop_h = crop_w * 2
                X1 = int((x1 + x2) / 2 - crop_w / 2)
                X2 = X1 + crop_w
                Y1 = int((y1 + y2) / 2 - crop_h / 3)
                Y2 = Y1 + crop_h
                ## adjust box size by image boundary
                crop_x1 = max(0, X1)
                crop_x2 = min(W-1, X2)
                crop_y1 = max(0, Y1)
                crop_y2 = min(H-1, Y2)
                cropped = frame[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]
                ## detect edge and text
                neck_line = y2 - crop_y1
                body_bound = int(p['body_bound'] * H) - crop_y1
                crop_y = detect_edge_text(cropped, neck_line)
                crop_y = min(crop_y, body_bound)
                cropped = cropped[:crop_y, :, :]        
                ratio = 1. * cropped.shape[0] / cropped.shape[1]
                ratio_matrix[index[0]].append(ratio)
#                 ## compare histogram to find obstacle
#                 y_med = int((neck_line + crop_y) / 2)
#                 img_bright = np.max(cropped, axis=2)
#                 hist_top = cv2.calcHist([img_bright[neck_line:y_med, :]], [0], None, [16], [0, 255])
#                 hist_bottom = cv2.calcHist([img_bright[y_med:, :]], [0], None, [16], [0, 255])
#                 ## normalization
#                 hist_top /= np.linalg.norm(hist_top)
#                 hist_bottom /= np.linalg.norm(hist_bottom)
#                 dist = np.linalg.norm(hist_top - hist_bottom)
#                 text = '{0:.3f}'.format(dist)
#                 cv2.rectangle(cropped, (0, 0), (cropped.shape[1]-1, 30), color=(255,255,255), thickness=-1)
#                 cv2.putText(cropped, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,0,0), thickness=2)
                p['frame'] = cropped
        fid += 1
    cap.release()
    
    res = []
    NUM_ANCHOR = 5
    folder = '../data/cloth_test'
    os.mkdir(folder)
    for i, anchor_person in enumerate(anchor_group):
        top_ratio = np.argsort(ratio_matrix[i])[::-1]
        num_anchor = min(NUM_ANCHOR, len(anchor_person))
        for j in range(num_anchor):
            filename = video_name + '_' + str(i) + '_' + str(j) + '.jpg'
            cv2.imwrite(os.path.join(folder, filename), anchor_person[top_ratio[j]]['frame'])
            res.append([video_name, i, filename])
    return res

def prepare_cloth_crop(video_name, anchor_group, storage):
    ## return [] if anchor_group is empty
    if len(anchor_group) == 0:
        return []

    ## collect face by fid
    fid2person = {}
    ratio_matrix = []
    for i, anchor_person in enumerate(anchor_group):
        for j, p in enumerate(anchor_person):
            if not p['fake']:
                ### for bug in hwang when loading frames at the end of the video
                if p['fid'] > 109500:
                    break
                ###
                if not p['fid'] in fid2person:
                    fid2person[p['fid']] = []
                p['index'] = (i, j)
                fid2person[p['fid']].append(p)
        ratio_matrix.append([])
    fid_list = sorted(fid2person)    
        
    ## load frames from google cloud
#     print("start loading frames...")
#     storage = StorageBackend.make_from_config(StorageConfig.make_gcs_config('esper'))
    video_path = 'tvnews/videos/' + video_name + '.mp4'
    video_file = RandomReadFile(storage, video_path.encode('ascii'))
    video = Decoder(video_file)
    img_list = video.retrieve(fid_list)
    img_list = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_list]
#     print("finish loading frames...")
    
    H, W = img_list[0].shape[:2]
    
    for i, fid in enumerate(fid_list):
        for p in fid2person[fid]:
            index = p['index']
            frame = img_list[i]
            x1 = int(p['bbox']['bbox_x1'] * W)
            y1 = int(p['bbox']['bbox_y1'] * H)
            x2 = int(p['bbox']['bbox_x2'] * W)
            y2 = int(p['bbox']['bbox_y2'] * H)
            ## set crop window
            crop_w = (x2 - x1) * 2
            crop_h = crop_w * 2
            X1 = int((x1 + x2) / 2 - crop_w / 2)
            X2 = X1 + crop_w
            Y1 = int((y1 + y2) / 2 - crop_h / 3)
            Y2 = Y1 + crop_h
            ## adjust box size by image boundary
            crop_x1 = max(0, X1)
            crop_x2 = min(W-1, X2)
            crop_y1 = max(0, Y1)
            crop_y2 = min(H-1, Y2)
            cropped = frame[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]
            ## detect edge and text
            neck_line = y2 - crop_y1
            body_bound = int(p['body_bound'] * H) - crop_y1
            crop_y = detect_edge_text(cropped, neck_line)
            crop_y = min(crop_y, body_bound)
            cropped = cropped[:crop_y, :, :]        
            ratio = 1. * cropped.shape[0] / cropped.shape[1]
            ratio_matrix[index[0]].append(ratio)
            p['frame'] = cropped

    res = []
    NUM_ANCHOR = 20
    folder = '../data/cloth_label'
    if not Path(folder).is_dir():
        os.mkdir(folder)
    for i, anchor_person in enumerate(anchor_group):
        top_ratio = np.argsort(ratio_matrix[i])[::-1]
        num_anchor = min(NUM_ANCHOR, len(top_ratio))
        for j in range(num_anchor):
            filename = video_name + '_' + '{:02}'.format(i) + '_' + '{:02}'.format(j) + '.jpg'
            cv2.imwrite(os.path.join(folder, filename), anchor_person[top_ratio[j]]['frame'])
            res.append([video_name, i, filename])
    return res

def solve_single_video(video_name, anchor_group, storage):
    res = prepare_cloth_crop(video_name, anchor_group, storage)
    return res

def solve_thread(anchor_dict_t, tmp_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    
    ## load dict
    res_dict = []
    meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    storage = StorageBackend.make_from_config(StorageConfig.make_gcs_config('esper'))
    
    i = 0
    video_list = sorted(anchor_dict_t)
    sample = np.random.choice(len(video_list), 40)
#     for video_name in sorted(anchor_dict_t):
    for idx in sample:
        video_name = video_list[idx]
        video_length = meta_dict[video_name]['video_length']
        if video_name in meta_dict and (video_length < 3000 or video_length > 4000):
            continue
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        res = solve_single_video(video_name, anchor_dict_t[video_name], storage)
        res_dict.extend(res)
#         if i % 1 == 0:
        pickle.dump(res_dict, open(tmp_dict_path, "wb"), protocol=2)
        i += 1
            
    pickle.dump(res_dict, open(tmp_dict_path, "wb"), protocol=2)
    print("Thread %d finished computing..." % (thread_id))

def solve_parallel(anchor_dict, res_dict_path=None, nthread=16, use_process=True):
    video_list = sorted(anchor_dict)
    
    ## remove exist video
#     dict_file = Path(res_dict_path)
#     if dict_file.is_file():
#         res_dict = pickle.load(open(res_dict_path, "rb" ))
#         video_list = [video for video in video_list if video not in res_dict]
#     else:
    res_dict = []

    num_video = len(video_list)
    print(num_video)
    if num_video == 0:
        return 
    if num_video <= nthread:
        nthread = num_video
        num_video_t = 1
    else:
        num_video_t = int(math.ceil(1. * num_video / nthread))
    print(num_video_t)
    
    tmp_dict_list = []
    for i in range(nthread):
        tmp_dict_list.append('../tmp/cloth_dict_' + str(i) + '.pkl')

#     if use_process:
#         ctx = mp.set_start_method('spawn')
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
#             video_list_t = video_list[0 : 50]
        else:
            video_list_t = video_list[i*num_video_t : ]
#             video_list_t = video_list[100 : 150]

        anchor_dict_t = {}
        for video in video_list_t:
            anchor_dict_t[video] = copy.deepcopy(anchor_dict[video])
            
        if use_process:
            t = mp.Process(target=solve_thread, args=(anchor_dict_t, tmp_dict_list[i], i,))
        else:
            t = threading.Thread(target=solve_thread, args=(anchor_dict_t, tmp_dict_list[i], i,))
            t.setDaemon(True)
        thread_list.append(t)
    
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    
    for path in tmp_dict_list:
        dict_file = Path(path)
        if not dict_file.is_file():
            continue
        res_dict_tmp = pickle.load(open(path, "rb" ))
#         res_dict = {**res_dict, **res_dict_tmp}
        res_dict.extend(res_dict_tmp)
    
    pickle.dump(res_dict, open(res_dict_path, "wb" ), protocol=2)  