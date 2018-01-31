from matplotlib import pyplot as plt
from utility import *
import copy
import pickle

def get_scores(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None, show_sentiment=False):
    topic_score = {'ALL':{}, 'CNN':{}, 'FOXNEWS':{}, 'MSNBC':{}}
    for s in topic_score:
        for t in topic:
            topic_score[s][t] = {}
    if show_sentiment:
        sentiment_score = copy.deepcopy(topic_score)
    
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
#                 if t in value[ttype]:
#                     cnt[t] += 1
                for i in range(len(value[ttype])):
                    if value[ttype][i] == t:
                        cnt[t] += (1.5 - 0.1*i)
                        break
                if show_sentiment:
                    senti[t].append(value['sentiment'][0])
        for t in topic:
            if not date in topic_score['ALL'][t]:
                topic_score['ALL'][t][date] = []
            if not date in topic_score[station][t]:
                topic_score[station][t][date] = []
            topic_score['ALL'][t][date].append(1. * cnt[t] / total/1.5)
            topic_score[station][t][date].append(1. * cnt[t] / total/1.5)
            
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
    
    return dates, topic_score_avg, sentiment_score_avg

def plot_video_list(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None, show_sentiment=False):
    dates, topic_score_avg, sentiment_score_avg = get_scores(topic_res, topic, ttype, start_date, end_date, show_name, station_name, show_sentiment)
    
    if show_name != None or station_name == None:
        station_name = ['ALL']
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
    
def filter_single_topic(topic_res, topic, ttype, start_date, end_date, show_name=None, station_name=None):
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
        
        for seg, value in topic_res[video_name].items():
#             if topic == value[ttype][0]:
            if topic in value[ttype]:
                if not video_name in filtered_topic_res:
                    filtered_topic_res[video_name] = {}
                filtered_topic_res[video_name][seg] = value
    return filtered_topic_res

def find_cooccurrence(topic_res, topic_query, type_query, start_date, end_date, show_name=None, station_name=None, top_k=20):
    if len(topic_query) > 0:
        filtered_topic_res = filter_single_topic(topic_res, topic_query, type_query, start_date, end_date, show_name, station_name)
    else:
        filtered_topic_res = topic_res
    
    # prepare topic_dict
    topic_dict = pickle.load(open("../data/topic_dict.pkl", 'rb'))
    topic_types = sorted(topic_dict)
    topic2id = {}
    for topic_type in topic_types:
        topic2id[topic_type] = {}
        idx = 0
        for t in topic_dict[topic_type]:
            topic2id[topic_type][t] = idx
            idx += 1
    
    # compute idf
    count_all = {}
    num_segs = 0
    for topic_type in topic_types:
        count_all[topic_type] = np.zeros(len(topic_dict[topic_type]))
    for video_name in sorted(topic_res):
        num_segs += len(topic_res[video_name])
        for seg, value in topic_res[video_name].items():
            for topic_type in topic_types:
                for t in value[topic_type]:
                    if t != None:
                        count_all[topic_type][topic2id[topic_type][t]] += 1
    # compute tf
    score = {}
    for topic_type in topic_types:
        score[topic_type] = np.zeros(len(topic_dict[topic_type]))
    for video_name in sorted(filtered_topic_res):
        for seg, value in filtered_topic_res[video_name].items():
            for topic_type in topic_types:
                for i in range(len(value[topic_type])):
                    t = value[topic_type][i]
                    if t == None or t == topic_query:
                        continue
                    score[topic_type][topic2id[topic_type][t]] += (1.5 - 0.1*i)
    # compute tf-idf
    for topic_type in topic_types:
        for i in range(len(count_all[topic_type])):
            if count_all[topic_type][i] == 0:
                count_all[topic_type][i] = num_segs
        score[topic_type] = score[topic_type] * np.log(1. * num_segs / count_all[topic_type])
    
    for topic_type in topic_types:
        top_id = np.argsort(score[topic_type])[::-1]
        fig = plt.figure()
        fig.set_size_inches(16, 7)
        score_plot = []
        topic_plot = []
        max_score = score[topic_type][top_id[0]]
        for i in range(top_k):
            score_plot.append(1. * score[topic_type][top_id[i]] / max_score)
            topic_plot.append(topic_dict[topic_type][top_id[i]])
        plt.bar(np.arange(top_k), score_plot)
        
        if topic_type == type_query:
            topic_plot_res = topic_plot
        
        plt.xticks(np.arange(top_k), topic_plot, rotation=60)
#         plt.xlabel('closely related '+ topic_type)
        plt.title('closely related '+ topic_type + ' to ' + topic_query)
    
    return topic_plot_res  

def plot_most_mentioned(topic_res, start_date, end_date):
    pop_subjects = ['politics', 'government', 'liberals', 'police', 'voters', 'debates', 'democracy', 'community', 'religion', 'children', 'nation', 'elections', 'legislation', 'money', 'family', 'primaries', 'law', 'military', 'crime', 'economy', 'conservatives', 'weather', 'terrorism', 'water', 'polls', 'speeches', 'leadership', 'race', 'governors', 'diplomacy']
    pop_phrases = ['White House', 'health care', 'national security', 'Republican Party', 'Vice President', 'foreign policy', 'Wall Street', 'Democratic Party', 'Obama administration', 'collusion', '9/11', 'tax returns', 'fake news', 'civil rights', 'high school', 'nuclear weapons', 'tax cuts', 'small business', 'Affordable Care Act', 'Second Amendment', 'health insurance', 'chemical weapons', 'Brexit', 'death penalty', 'minimum wage', 'human rights', 'Civil War', 'sexual assault', 'mental health', 'stock market']
    pop_people = ['Trump, Donald', 'Clinton, Hillary', 'Sanders, Bernie', 'Cruz, Ted', 'Clinton, Bill', 'Bush, Jeb', 'Obama, Barack', 'Rubio, Marco', 'Comey, James', 'Putin, Vladimir', 'Ryan, Paul', 'Sessions, Jeff', 'Flynn, Michael', 'Moore, Roy', 'Romney, Mitt', 'Pence, Mike', 'McConnell, Mitch', 'McCain, John', 'Biden, Joe', 'Kushner, Jared', 'Christie, Chris', 'Kasich, John', 'Spicer, Sean', 'Paul, Rand', 'Mueller, Robert', 'Tillerson, Rex', 'Manafort, Paul', 'Trump, Ivanka', 'Reagan, Ronald', 'Trump, Melania']
    stations = ['CNN', 'FOXNEWS', 'MSNBC']    
    N = 30
    dates, topic_score_avg, sentiment_score_avg = get_scores(topic_res, pop_people[:N], 'people', start_date, end_date, station_name=stations, show_sentiment=False)
#     for s in stations:
#         print(s, np.average(sentiment_score_avg[s]['politics']))  
#     return sentiment_score_avg

    width = 0.27
    x = np.arange(N)
    topic_plot = pop_people[:N]
    bars = []
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    for i in range(len(stations)):
        s = stations[i]
        y = []
        for t in topic_plot:
            y.append(np.average(topic_score_avg[s][t]))
        bar = plt.bar(x+width*i, y, width)
        bars.append(bar)
    plt.xticks(x+width, topic_plot, rotation=45)
    plt.legend( (bars[0][0], bars[1][0], bars[2][0]), (stations[0], stations[1], stations[2]) )
    plt.show()
        