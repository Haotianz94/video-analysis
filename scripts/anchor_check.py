from sklearn.cluster import KMeans
from utility import *

def build_anchor_center(anchor_dict):
    anchor_center_dict = {}
    for video, anchor_group in anchor_dict.items():
        anchor_center_dict[video] = []
        for anchor_person in anchor_group:
            for p in anchor_person:
                if p['sim'] == 0:
                    anchor_center_dict[video].append(p)
                    break
    return anchor_center_dict

def group_by_show(random_dict):
    dict_by_show = {}
    for video, group in random_dict.items():
        data, station, show = get_detail_from_video_name(video)
        if not show in dict_by_show:
            dict_by_show[show] = {}
        dict_by_show[show][video] = group
    return dict_by_show

def group_by_station(random_dict):
    dict_by_station = {}
    for video, group in random_dict.items():
        data, station, show = get_detail_from_video_name(video)
        if not station in dict_by_station:
            dict_by_station[station] = {}
        dict_by_station[station][video] = group
    return dict_by_station

def cluster_real_anchor(anchor_center):
    ## cluster from centers
#     anchor_center = build_anchor_center(anchor_dict)
    
    num_videos = len(anchor_center)
    features = []
    for video, centers in anchor_center.items():
        for p in centers:
            features.append(p['feature'])
    NUM_CLUSTER = 10
#     print(len(features))
    kmeans = KMeans(n_clusters=NUM_CLUSTER).fit(features)
    
    cluster_center = []
    cluster = []
    for k in range(NUM_CLUSTER):
        cluster_center.append(kmeans.cluster_centers_[k])
        cluster.append([])
    for j, c in enumerate(kmeans.labels_):
        cluster[c].append(j)
        
    ## merge similar clusters
    MERGE_THRESH = 0.6
    while True:
        del_key = -1
        for i, c1 in enumerate(cluster_center):
            if del_key != -1:
                break
            for j in range(i+1, len(cluster_center)):
                c2 = cluster_center[j]
                dist = np.linalg.norm(c1 - c2)
                if dist < MERGE_THRESH:
                    if len(cluster[i]) < len(cluster[j]):
                        cluster[j].extend(cluster[i])
                        del_key = i
                    else:
                        cluster[i].extend(cluster[j])
                        del_key = j
                    break
        if del_key != -1:
            del cluster[del_key]
            del cluster_center[del_key]
        else:
            break    
    
    cluster_cnt = []
    for k in range(len(cluster)):
        cluster_cnt.append(len(cluster[k]))
    sort_idx = np.argsort(cluster_cnt)[::-1]
    
    ANCHOR_THRESH = 0.5
    real_anchor = []
    for idx in sort_idx:
#         print(1. * cluster_cnt[idx] / num_videos)
        if 1. * cluster_cnt[idx] / num_videos > ANCHOR_THRESH:
            real_anchor.append(cluster_center[idx])
            
    return real_anchor        

def check_non_anchor_by_show(anchor_dict, show_name):
    
    anchor_dict_show = group_by_show(anchor_dict)
    real_anchor = cluster_real_anchor(anchor_dict_show[show_name])
    anchor_center = build_anchor_center(anchor_dict_show[show_name])
    ## check cases when anchor are not showing
    for video, centers in anchor_center.items():
        has_real_anchor = np.zeros(len(real_anchor))
        for center in centers:
            for ra_idx, ra in enumerate(real_anchor):
                if np.linalg.norm(ra - center['feature']) < MERGE_THRESH:
                    has_real_anchor[ra_idx] += 1
                    break
        non_zero = np.count_nonzero(has_real_anchor)
        if non_zero == 0:
            print("No real anchor: %s" % video)
        elif non_zero < len(real_anchor):
            print("Not enough real anchor: %s" % video)
        elif non_zero > len(real_anchor):
            print("To much real anchor: %s" % video)

def split_dataset(data, meta, labels, test_ratio=0.2):
    N = len(data)
    test_idx = np.random.choice(int(test_ratio * N), N)
    data_train = []
    data_test = []
    meta_train = []
    meta_test = []
    labels_train = []
    labels_test = []
    for idx in range(N):
        if idx in test_idx:
            data_test.append(data[idx])
            meta_test.append(meta[idx])
            labels_test.append(labels[idx])
        else:
            data_train.append(data[idx])
            meta_train.append(meta[idx])
            labels_train.append(labels[idx])
    return data_train, meta_train, labels_train, data_test, meta_test, labels_test            
            