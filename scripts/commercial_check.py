from matplotlib import pyplot as plt
from utility import *
import numpy as np
import pickle
import pysrt
from pathlib import Path
import codecs
from commercial_search import *

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
def visualize_video_single(commercial_list, video_desp, groundtruth=None, raw_commercial_list=None, lowertext_window_list=None, blanktext_window_list=None):
    y_gt = [1, 1]
    y_com = [2, 2]
    y_raw = [3, 3]
    y_lower = [4, 4]
    y_blank = [5, 5]
    fig = plt.figure()
    
    if not groundtruth is None: 
        for gt in groundtruth:
            text = 'gt:  ' + str(gt[0]) + '-' + str(gt[1])
            plt.plot([get_second(gt[0]), get_second(gt[1])], y_gt, 'g', linewidth=2.0, label=text)

    for com in commercial_list:
        text = 'our: ' + str(com[0][1]) + '-'+ str(com[1][1])
        plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_com, 'r', linewidth=2.0, label=text)
    
    if not raw_commercial_list is None: 
        for com in raw_commercial_list:
            text = 'raw: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_raw, 'y', linewidth=2.0, label=text)

    if not lowertext_window_list is None:
        for com in lowertext_window_list:
            text = 'lower: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_lower, 'b', linewidth=2.0, label=text)
    
    if not blanktext_window_list is None:
        for com in blanktext_window_list:
            text = 'blank: ' + str(com[0][1]) + '-'+ str(com[1][1])
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], y_blank, 'k', linewidth=2.0, label=text)
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([0, 10])
    plt.xlim([0, video_desp['video_length']])
    plt.xlabel('video time (s)')
    cur_axes = plt.gca()
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()

def visualize_video_list(result, commercial_gt, video_length):
    
    fig = plt.figure()
    fig.set_size_inches(14, 30)
    ax = fig.add_subplot(111)
    vid = 1
    for video_name in sorted(result):
            
        commercial_list = result[video_name]['commercial']
        for com in commercial_list:
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], [vid, vid], 'r', linewidth=1.0)
        if video_name in commercial_gt:
            groundtruth = commercial_gt[video_name]['span']
            for gt in groundtruth:
                plt.plot([get_second(gt[0]), get_second(gt[1])], [vid-0.2, vid-0.2], 'g', linewidth=1.0)
        
        ax.text(-1500, vid, video_name)
        ax.text(video_length, vid, str(result[video_name]['commercial_length']))

        vid += 1
    
    # draw vertical segment
    seg = 500
    while seg < video_length:
        plt.plot([seg, seg], [0, vid], 'k', linewidth=0.7)
        seg += 500
    
    
#     legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([0, vid])
    plt.xlim([0, video_length])
    plt.xlabel('video time (s)')
    cur_axes = plt.gca()
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()
    
def find_popular_location(result, video_length):
    seg_pos = []
    hit_num = []
    seg = 30.0
    cur = 0
    while cur < video_length:
        hit = 0
        for key, value in result.items():
            commercial_list = value['commercial']
            for com in commercial_list:
                start = get_second(com[0][1])
                end = get_second(com[1][1])
                if cur >= start and cur <= end:
                    hit += 1
                    break
        seg_pos.append(cur / 60)
        hit_num.append(hit)
        cur += seg
    
    plt.figure()
    plt.bar(seg_pos, hit_num)
    plt.xlabel('video time (s)')
    plt.ylabel('number of commercials')
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
    return (precision, recall)

def detect_suspicous(result):
    MIN_SPAN_THRESH = 60
    MAX_SPAN_THRESH = 270
    MIN_GAP_THRESH = 60
    NUM_MAX_THRESH = 6
    NUM_MIN_THRESH = 3
    count = 0
    suspicous_video = []
    long_video = []
    all_spans = []
    
    for video_name in sorted(result):
        print(video_name)
        suspicous = False
        commercial_list = result[video_name]['commercial']
        for i in range(len(commercial_list)):
            com = commercial_list[i]
            span = get_time_difference(com[0][1], com[1][1])
            if span < MIN_SPAN_THRESH:
                print(com[0][1], com[1][1], "[small block]")
                suspicous = True
            elif span > MAX_SPAN_THRESH:
                print(com[0][1], com[1][1], "[large block]")
                suspicous = True
                long_video.append(video_name)
            else:
                print(com[0][1], com[1][1])
            all_spans.append(span)    
                
        num_com = len(commercial_list)
        if num_com > NUM_MAX_THRESH:
            print("[too many blocks]")
            suspicous = True
        if num_com < NUM_MIN_THRESH:
            print("[too few blocks]")
            suspicous = True
#             if i != len(commercial_list)-1:
#                 com_next = commercial_list[i+1]
#                 gap = get_time_difference(com[1][1], com_next[0][1])
#                 if gap < MIN_GAP_THRESH:
#                     print('[small gap]')
#                     suspicous = True
        if suspicous:
            suspicous_video.append(video_name)
            count += 1
    print("suspicous videos: %d" % count)
    
    for video_name in suspicous_video:
        print(video_name)
        
    return all_spans, long_video            

def get_stat_from_result():
    commercial_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    # group all the videos by show name
    show_group_stat = {}
    for video_name in sorted(commercial_dict):
        station_name = video_name.split('_')[0]
        if station_name == 'CNNW':
            show_name = video_name[21:]
        elif station_name == 'FOXNEWSW':
            show_name = video_name[25:]
        elif station_name == 'MSNBCW':
            show_name = video_name[23:]
        if not show_name in show_group_stat:
            show_group_stat[show_name] = {}
            show_group_stat[show_name]['videos'] = {}
        show_group_stat[show_name]['videos'][video_name] = {}
        show_group_stat[show_name]['videos'][video_name]['commercial_list'] = commercial_dict[video_name]
        show_group_stat[show_name]['videos'][video_name]['video_length'] = video_meta_dict[video_name]['video_length']

    # count commercial block numbers
    cnt = 0
    show_com_counts = {}
    for show_name in sorted(show_group_stat):
        com_counts = []
        for video_name in sorted(show_group_stat[show_name]['videos']):
            num_hour = np.round(1. * show_group_stat[show_name]['videos'][video_name]['video_length'] / 3600)
            if num_hour == 0:
                num_hour = 1
            coms = len(show_group_stat[show_name]['videos'][video_name]['commercial_list']) / num_hour
            if coms > 7:
                cnt += 1
#                 print(video_name, num_hour, coms)
            com_counts.append(coms)
            show_group_stat[show_name]['videos'][video_name]['commercial_num'] = coms
        show_com_counts[show_name] = np.array(com_counts)
        show_group_stat[show_name]['com_counts'] = np.array(com_counts)
#     print(cnt)
    # calculate commercial ratio
    show_com_cvgs = {}
    for show_name in sorted(show_group_stat):
        com_cvgs = []
        for video_name in sorted(show_group_stat[show_name]['videos']):
            com_len = 0
            for com in show_group_stat[show_name]['videos'][video_name]['commercial_list']:
                com_len += get_time_difference(com[0][1], com[1][1])
            cvg = 1. * com_len / video_meta_dict[video_name]['video_length']
#             if cvg > 0.5:
#                 print(video_name, com_len, video_meta_dict[video_name]['video_length'])
#             cvg = float("{0:.3f}".format(cvg))
            com_cvgs.append(cvg)
            show_group_stat[show_name]['videos'][video_name]['commercial_ratio'] = cvg
        show_com_cvgs[show_name] = np.array(com_cvgs)
        show_group_stat[show_name]['com_cvgs'] = np.array(com_cvgs)

    
    for show_name in sorted(show_group_stat):
        show_group_stat[show_name]['stat_num'] = {}
        show_group_stat[show_name]['stat_num']['max'] = np.max(show_com_counts[show_name])
        show_group_stat[show_name]['stat_num']['min'] = np.min(show_com_counts[show_name])
        show_group_stat[show_name]['stat_num']['avg'] = np.average(show_com_counts[show_name])
        show_group_stat[show_name]['stat_num']['std'] = np.std(show_com_counts[show_name])
        show_group_stat[show_name]['stat_num']['median'] = np.median(show_com_counts[show_name])
        show_group_stat[show_name]['stat_num']['num'] = show_com_counts[show_name].shape[0]
        show_group_stat[show_name]['stat_ratio'] = {}
        show_group_stat[show_name]['stat_ratio']['max'] = np.max(show_com_cvgs[show_name])
        show_group_stat[show_name]['stat_ratio']['min'] = np.min(show_com_cvgs[show_name])
        show_group_stat[show_name]['stat_ratio']['avg'] = np.average(show_com_cvgs[show_name])
        show_group_stat[show_name]['stat_ratio']['std'] = np.std(show_com_cvgs[show_name])
        show_group_stat[show_name]['stat_ratio']['median'] = np.median(show_com_cvgs[show_name])
    return show_group_stat, commercial_dict

def check_transcript_ratio():
    ratio_dict = pickle.load(open('../data/ratio_dict.pkl','rb'))
    video_meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    cnt = 0
    for line in open('../data/total_video_list.txt', 'r'):
        video_name = line[:-1]
        if video_name in ratio_dict:
            continue
        srt_path = '../data/transcripts/' + video_name + '.cc5.srt'

        srt_file = Path(srt_path)
        if not srt_file.is_file():
            srt_path = srt_path.replace('cc5', 'cc1')
            srt_file = Path(srt_path)
            if not srt_file.is_file():
                srt_path = srt_path.replace('cc1', 'align')
                srt_file = Path(srt_path)
                if not srt_file.is_file():
                    continue

        try:
            file = codecs.open(srt_path, encoding='utf-8', errors='strict')
            for line in file:
                pass
        except UnicodeDecodeError:
            continue

        subs = pysrt.open(srt_path)
        text_length = 0
        for sub in subs:
            text_length += get_time_difference(tuple(sub.start)[:3], tuple(sub.end)[:3])            
        ratio_dict[video_name] = 1. * text_length / video_meta_dict[video_name]['video_length']
        
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        
    pickle.dump(ratio_dict, open('../data/ratio_dict2.pkl', 'wb'))
    return ratio_dict
    
def visualize_commercials_per_minute():
    commercial_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    counts = np.zeros(60)
    time = np.arange(60)
    
    for video in commercial_dict:
        clist = commercial_dict[video]
        for com in clist:
            start = int(get_second(com[0][1]) / 60)
            end = int(get_second(com[1][1]) / 60)
            if start >= 60:
                continue
            if end >= 60:
                end = 59
            for i in range(start, end+1):
                counts[i] += 1
                
    plt.figure()
    plt.bar(time, counts)
    plt.xlabel('video time (min)')
    plt.ylabel('number of commercials hit')
    plt.show()

def visualize_show(show_group_stat, show_name):
    show_group = show_group_stat[show_name]
    
    counts = np.zeros(60)
    time = np.arange(60)
    for video in show_group['videos']:
        commercial_list = show_group['videos'][video]['commercial_list']
        for com in commercial_list:
            start = int(get_second(com[0][1]) / 60)
            end = int(get_second(com[1][1]) / 60)
            if start >= 60:
                continue
            if end >= 60:
                end = 59
            for i in range(start, end+1):
                counts[i] += 1
    
    plt.figure(1)
    plt.bar(time, counts)
    plt.xlabel('video time (min)')
    plt.ylabel('number of commercials hit')
    plt.title(show_name)
#     plt.show()
    plt.savefig('../result/fig/'+show_name+'_dis.png')

    max_com = int(np.max(show_group['com_counts']))    
    y = np.bincount(show_group['com_counts'].astype(int))
    x = np.arange(max_com+1)
    y = y / show_group['com_counts'].shape[0] * 100
    plt.figure(2)
    plt.bar(x, y)
    plt.xlabel('number of commercials blocks per hour')
    plt.ylabel('percent of videos (%)')
    plt.title(show_name)
#     plt.show()  
    plt.savefig('../result/fig/'+show_name+'_num.png')
    
    num_seg = 1000
    counts = np.zeros(num_seg)
    for cvg in show_group['com_cvgs']:
        seg = int(cvg * num_seg)
        counts[seg] += 1
    counts = counts / show_group['com_cvgs'].shape[0] * 100
    plt.figure(3)
    plt.bar(np.arange(num_seg) / num_seg * 100, counts)
    plt.xlabel('commercial coverage (%)')
    plt.ylabel('percent of videos (%)')
    plt.title(show_name)
#     plt.show()  
    plt.savefig('../result/fig/'+show_name+'_cvg.png')

def check_com_shot_align(com_dict=None, shot_dict=None, com_dict_path='../data/commercial_dict.pkl', shot_dict_path='../data/shot_dict.pkl', num_visualize=100):
    if com_dict == None:
        com_dict = pickle.load(open(com_dict_path, 'rb'))
    if shot_dict == None:
        shot_dict = pickle.load(open(shot_dict_path, 'rb'))
    meta_dict = pickle.load(open('../data/video_meta_dict.pkl', 'rb'))
    commercial_gt = load_commercial_groundtruth()
    
    randselect = np.random.choice(len(com_dict), num_visualize)
    video_list = sorted(com_dict)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vid = 1
#     for i in randselect:
#         video_name = video_list[i]
    for video_name in commercial_gt:
        if not video_name in shot_dict or not video_name in com_dict:
            continue
        shot_list = shot_dict[video_name]
        fps = meta_dict[video_name]['fps']
        for shot in shot_list:
            x1 = get_second_from_fid(shot[0], fps)
            plt.plot([x1, x1], [vid-0.2, vid+0.2], 'b', linewidth=0.2)
        
        commercial_list = com_dict[video_name]
        for com in commercial_list:
            plt.plot([get_second(com[0][1]), get_second(com[1][1])], [vid, vid], 'r', linewidth=5.0)
        
        groundtruth = commercial_gt[video_name]['span']
        for gt in groundtruth:
            plt.plot([get_second(gt[0]), get_second(gt[1])], [vid-0.2, vid-0.2], 'g', linewidth=5.0)
        
        ax.text(-800, vid, video_name[:24])
        vid += 1
        print(vid)
        if vid > num_visualize:
            break
    
#     legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.set_size_inches(14, 40)
    plt.ylim([0, vid])
    plt.xlim([0, 3700])
    plt.xlabel('video time (s)')
    cur_axes = plt.gca()
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()
        
    
            