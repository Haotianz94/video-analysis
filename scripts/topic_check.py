from matplotlib import pyplot as plt
from utility import *

def plot_topic(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None):
    topic_score = {'ALL':{}, 'CNN':{}, 'FOXNEWS':{}, 'MSNBC':{}}
    for s in topic_score:
        for t in topic:
            topic_score[s][t] = {}
    
    for video_name in sorted(topic_res):
        split = video_name.split('_')
        date = get_date_from_string(split[1])
        if compare_date(start_date, date) > 0 or compare_date(date, end_date) > 0:
            continue
        station = split[0][:-1]
        if station == 'CNN':
            show = video_name[21:]
        elif station == 'FOXNEWS':
            show = video_name[25:]
        elif station == 'MSNBC':
            show = video_name[23:]
        if show_name != None and show != show_name:
            continue
        
        total = len(topic_res[video_name])
        cnt = {}
        for t in topic:
            cnt[t] = 0
        if ttype == 'subject':
            for t in topic:
                for seg, value in topic_res[video_name].items():
                    for sub in value['subject']:
                        if t in sub:
                            cnt[t] += 1
                            break
            for t in topic:
                if not date in topic_score['ALL'][t]:
                    topic_score['ALL'][t][date] = []
                if not date in topic_score[station][t]:
                    topic_score[station][t][date] = []
                topic_score['ALL'][t][date].append(1. * cnt[t] / total)
                topic_score[station][t][date].append(1. * cnt[t] / total)
        else:
            for t in topic:
                for seg, value in topic_res[video_name].items():
                    if t in value[ttype]:
                        cnt[t] += 1
            for t in topic:
                if not date in topic_score['ALL'][t]:
                    topic_score['ALL'][t][date] = []
                if not date in topic_score[station][t]:
                    topic_score[station][t][date] = []
                topic_score['ALL'][t][date].append(1. * cnt[t] / total)
                topic_score[station][t][date].append(1. * cnt[t] / total)
    
    dates = sorted(topic_score['ALL'][topic[0]])
    topic_score_avg = {}
    if show_name != None or station_name == None:
        station_name = ['ALL']
    for s in station_name:
        topic_score_avg[s] = {}
        for t in topic:
            topic_score_avg[s][t] = []
            for date in dates:
                if date in topic_score[s][t]:
                    topic_score_avg[s][t].append(np.average(topic_score[s][t][date]))
                else:
                    topic_score_avg[s][t].append(0)

    # plot
    x = np.arange(len(dates))
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    handles = []
    for s in station_name:
        for t in topic:
            if show_name != None:
                label_text = '\''+ t + '\' in ' + show_name
            else:
                label_text = '\''+ t + '\' in ' + s
            curve, = plt.plot(x, topic_score_avg[s][t], label=label_text)
            handles.append(curve)
    plt.legend(handles=handles)

    NUM_XTICKS = 36
    sample = int(len(x)/NUM_XTICKS)
    x_ticks = [x[i] for i in range(0, len(x), sample)]
    dates_ticks = [dates[i] for i in range(0, len(x), sample)]
    plt.xticks(x_ticks, dates_ticks, rotation = 90 )
    if show_name != None:
        show_station_str = 'in ' + show_name
    else:
        show_station_str = 'in '
        for s in station_name:
            show_station_str += (s+' ') 
    plt.title('Trend of %s from %d-%02d-%02d to %d-%02d-%02d %s'  \
        % (topic, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], show_station_str))
    plt.show()