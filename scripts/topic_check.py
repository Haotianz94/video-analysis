from matplotlib import pyplot as plt
from utility import *

def plot_topic(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None):
    # {(2017, 09, 05): [0.1,0.2,0.3]}
    topic_score = {}
    for video_name in sorted(topic_res):
        split = video_name.split('_')
        date = get_date_from_string(split[1])
        if compare_date(start_date, date) > 0 or compare_date(date, end_date) > 0:
            continue
        station = split[0]
        if station_name != None and station != station_name:
            continue
        if station == 'CNNW':
            show = video_name[21:]
        elif station == 'FOXNEWSW':
            show = video_name[25:]
        elif station == 'MSNBCW':
            show = video_name[23:]
        if show_name != None and show != show_name:
            continue
        total = len(topic_res[video_name])
        cnt = 0
        if ttype == 'subject':
            for seg, value in topic_res[video_name].items():
                for sub in value['subject']:
                    if topic in sub:
                        cnt += 1
                        break
            if not date in topic_score:
                topic_score[date] = []
            topic_score[date].append(1. * cnt / total)
        else:
            for seg, value in topic_res[video_name].items():
                if topic in value[ttype]:
                    cnt += 1
            if not date in topic_score:
                topic_score[date] = []
            topic_score[date].append(1. * cnt / total)
        
    dates = sorted(topic_score)
    topic_score_avg = []
    for date in dates:
        topic_score_avg.append(np.average(topic_score[date]))
    x = np.arange(len(dates))
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    plt.plot(x, topic_score_avg)
    
    NUM_XTICKS = 36
    sample = int(len(x)/NUM_XTICKS)
    x_ticks = [x[i] for i in range(0, len(x), sample)]
    dates_ticks = [dates[i] for i in range(0, len(x), sample)]
    plt.xticks(x_ticks, dates_ticks, rotation = 90 )
    show_station_str = 'in all news'
    if show_name != None:
        show_station_str = 'in ' + show_name
    if station_name != None:
        show_station_str = 'in ' + station_name
    plt.title('Trend of \"%s\" from %d-%02d-%02d to %d-%02d-%02d %s'  \
        % (topic, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], show_station_str))
    plt.show()