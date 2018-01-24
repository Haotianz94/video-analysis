from matplotlib import pyplot as plt
from utility import *
import copy

def plot_video_list(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None, show_sentiment=False):
    topic_score = {'ALL':{}, 'CNN':{}, 'FOXNEWS':{}, 'MSNBC':{}}
    for s in topic_score:
        for t in topic:
            topic_score[s][t] = {}
    if show_sentiment:
        sentiment_score = copy.deepcopy(topic_score)
    filtered_topic_res = {}
    
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
        senti = {}
        for t in topic:
            cnt[t] = 0
            senti[t] = []
            for seg, value in topic_res[video_name].items():
                if t in value[ttype]:
                    cnt[t] += 1
                    if not video_name in filtered_topic_res:
                        filtered_topic_res[video_name] = {}
                    filtered_topic_res[video_name][seg] = value
                if show_sentiment:
                    senti[t].append(value['sentiment'][0])
        for t in topic:
            if not date in topic_score['ALL'][t]:
                topic_score['ALL'][t][date] = []
            if not date in topic_score[station][t]:
                topic_score[station][t][date] = []
            topic_score['ALL'][t][date].append(1. * cnt[t] / total)
            topic_score[station][t][date].append(1. * cnt[t] / total)
            
            if show_sentiment:
                if not date in sentiment_score['ALL'][t]:
                    sentiment_score['ALL'][t][date] = []
                if not date in sentiment_score[station][t]:
                    sentiment_score[station][t][date] = []
                sentiment_score['ALL'][t][date].append(np.average(senti[t]))
                sentiment_score[station][t][date].append(np.average(senti[t]))
                
    dates = sorted(topic_score['ALL'][topic[0]])
    topic_score_avg = {}
    sentiment_score_avg = {}
    if show_name != None or station_name == None:
        station_name = ['ALL']
    for s in station_name:
        topic_score_avg[s] = {}
        if show_sentiment:
            sentiment_score_avg[s] = {}
        for t in topic:
            topic_score_avg[s][t] = []
            if show_sentiment:
                sentiment_score_avg[s][t] = []
            for date in dates:
                if date in topic_score[s][t]:
                    topic_score_avg[s][t].append(np.average(topic_score[s][t][date]))
                else:
                    topic_score_avg[s][t].append(0)
                if show_sentiment:
                    if date in sentiment_score[s][t]:
                        sentiment_score_avg[s][t].append(np.average(sentiment_score[s][t][date]))
                    else:
                        sentiment_score_avg[s][t].append(0)

    # plot
    x = np.arange(len(dates))
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    handles = []
    for s in station_name:
        for t in topic:
            if show_name != None:
                label_topic = '\''+ t + '\' in ' + show_name
                label_senti = 'sentiment of \''+ t + '\' in ' + show_name
            else:
                label_topic = '\''+ t + '\' in ' + s
                label_senti = 'sentiment of \''+ t + '\' in ' + s
            label_topic += "  Avg = " + '{0:.4f}'.format(np.average(topic_score_avg[s][t]))    
            curve1, = plt.plot(x, topic_score_avg[s][t], label=label_topic)
            handles.append(curve1)
            if show_sentiment:
                curve2, = plt.plot(x, sentiment_score_avg[s][t], label=label_senti)
                handles.append(curve2)
#             print("%s: average = %3f median = %3f" % (label_topic, np.average(topic_score_avg[s][t]), np.median(topic_score_avg[s][t])))    
                
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
    
    return filtered_topic_res

def plot_single_video(topic_list, topic, subject):
    colors = ['', 'b', 'c', 'm', 'y', 'k', 'r', 'g']
    # plot topic range
    fig = plt.figure(1)
#     fig.set_size_inches(14, 30)
    ax = fig.add_subplot(111)
    y_pos = 1
    for t in topic:
        for seg in sorted(topic_list):
            if t in topic_list[seg][subject]:
                plt.plot([seg[0], seg[1]], [y_pos, y_pos], colors[y_pos])
        ax.text(-1000, y_pos, t)
        y_pos += 1
    
    # plot sentiment
    time = []
    polarity = []
    subjectivity = []
    for seg in sorted(topic_list):
        time.append((seg[0]+seg[1])/2.0)
        polarity.append(topic_list[seg]['sentiment'][0])
        subjectivity.append(topic_list[seg]['sentiment'][1])
    plt.plot(time, polarity, 'r')
    plt.plot(time, subjectivity, 'g')
        
    plt.ylim([-1, y_pos])
#     plt.xlim([0, video_length])
    plt.xlabel('video time (s)')
#     cur_axes = plt.gca()
#     cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()