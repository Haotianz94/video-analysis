import numpy as np
import pysrt
import gensim
import textacy
import pickle
import time
from pathlib import Path
import codecs
import math
import threading
# import xml.etree.ElementTree as ET
from utility import *


def load_transcript(srt_path, w2v_model, SEGTIME):
    # Load transcripts
    subs = pysrt.open(srt_path)
    transcript = []
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))
    
    # split transcript into segment
    replace_list = ['\n', '>', '.', ',', '?', '!', '\'', '"', '-', '(', ')']
    TRANSCRIPT_DELAY = 6
    text_seg = []
    text = ''
    seg_end = SEGTIME
    for t in transcript:
        if get_second(t[1]) - TRANSCRIPT_DELAY < seg_end:
            text += t[0] + ' '
        else:
            for token in replace_list:
                text = text.replace(token, ' ')
            text = text.lower()
            text_seg.append(text)
            text = ''
            seg_end += SEGTIME
    
    # Extract PROPN and NOUN using textacy
    docs = [textacy.doc.Doc(text, lang=u'en') for text in text_seg]
    corpus = textacy.Corpus(u'en', docs=docs)

    text_seg_words = [list(textacy.extract.words(doc, filter_nums=True, include_pos=['PROPN', 'NOUN'])) for doc in corpus.docs]
    # remove word not in w2v model
    text_seg_words = [ [str(word) for word in seg if len(str(word)) > 1 and str(word) in w2v_model.vocab] for seg in text_seg_words]
    
    return text_seg, text_seg_words

def load_subject_from_meta(meta_path, w2v_model):
    # Extract keywords from video meta
    tree = ET.parse(meta_path)
    root = tree.getroot()
    subject_str = root.find('subject').text
    subject = subject_str.split(';')

    docs_subject = [textacy.doc.Doc(text, lang=u'en') for text in subject]
    corpus_subject = textacy.Corpus(u'en', docs=docs_subject)
    keywords = [list(textacy.extract.words(doc, filter_nums=True, include_pos=['PROPN', 'NOUN'])) for doc in corpus_subject.docs]
    # remove word not in w2v model
    keywords = [ [str(word) for word in seg if str(word) in w2v_model.vocab] for seg in keywords]
    # remove empty subject
    keywords = [words for words in keywords if words != []]
    # remove repeated word
    i = 0
    while i < len(keywords):
        delete = False
        for j in range(i):
            if keywords[i] == keywords[j]:
                del keywords[i]
                delete = True
                break
        if not delete:
            i += 1
    # Todo: remove one being subset of another
    return keywords

def load_topic_from_dict(w2v_model):
    # Extract keywords from subject dict
    topic_dict = pickle.load(open("../data/topic_dict.pkl", 'rb'))
    subject = topic_dict['subject']

    docs_subject = [textacy.doc.Doc(text, lang=u'en') for text in subject]
    corpus_subject = textacy.Corpus(u'en', docs=docs_subject)
    keywords = [list(textacy.extract.words(doc, filter_nums=True, include_pos=['PROPN', 'NOUN'])) for doc in corpus_subject.docs]
    # remove word not in w2v model
    keywords = [ [str(word) for word in seg if str(word) in w2v_model.vocab] for seg in keywords]
    # remove empty subject
    keywords = [words for words in keywords if words != []]
    # remove repeated word
    i = 0
    while i < len(keywords):
        delete = False
        for j in range(i):
            if keywords[i] == keywords[j]:
                del keywords[i]
                delete = True
                break
        if not delete:
            i += 1
    return topic_dict, keywords

# help function
def count_name(text, name):
    cnt = 0
    start = -1
    ln = len(name)
    while True:
        start = text.find(name, start + 1)
        if start == -1:
            break
        else:
            if text[start+ln] != ':' and not text[start+ln].isalpha():
                cnt += 1
    return cnt

def assign_topic_detail(text_seg, text_seg_words, topic_dict, keywords, w2v_model, SEGTIME):
    num_seg = len(text_seg_words)
    num_key = len(keywords)
    topic_list = {}
    for i in range(num_seg):
        print('Segment: %d-%d min\n' % (i*SEGTIME/60, (i+1)*SEGTIME/60))
        # remove commercial seg
        if len(text_seg_words[i]) < SEGTIME/60*10:
            continue

        seg_index = (i*SEGTIME, (i+1)*SEGTIME)
        topic_list[seg_index] = {}
        # topic: subject
        print("### topic: subject ###")
        topic_list[seg_index]['subject'] = []
        sim_matrix = np.zeros(num_key)
        for j in range(num_key):
            avg_sim_key = []
            for key in keywords[j]:
                if text_seg_words[i] != []:
                    sim = [w2v_model.wv.similarity(word, key) for word in text_seg_words[i]]
                    avg_sim_key.append(np.average(sim))
                else:
                    avg_sim_key.append(0)
            sim_matrix[j] = np.average(avg_sim_key)

        topic_id = sim_matrix.argsort()[::-1]
        for j in range(5):
            print(sim_matrix[topic_id[j]], keywords[topic_id[j]])
            topic_list[seg_index]['subject'].append(keywords[topic_id[j]])
        print("\ntopic related words:")
        key = keywords[topic_id[0]][0]
        sim = np.array([w2v_model.wv.similarity(word, key) for word in text_seg_words[i]])
        sim_word_id = sim.argsort()[::-1]
        sim_words = [text_seg_words[i][sim_word_id[j]] for j in range(20) if j < len(text_seg_words[i])]
        print(sim_words)

        # topic: people 
        LAST_SPECIAL = {'donald trump', 'hillary clinton', 'barack obama'}
        print("\n### topic: people ###")
        topic_list[seg_index]['people'] = []
        people_count = []
        for people in topic_dict['people']:
            names = people.split(',')
            if len(names) == 1:
                cnt = text_seg[i].count(names[0].lower())
            else:
                lastname = names[0].lower()
                firstname = names[1].lower()
                last_cnt = count_name(text_seg[i], lastname)
                full_cnt = count_name(text_seg[i], firstname+' '+lastname)
                if full_cnt == 0:
                    if firstname+' '+lastname in LAST_SPECIAL:
                        cnt = last_cnt
                    else:
                        cnt = 0
                else:
                    cnt = last_cnt + full_cnt
            people_count.append(cnt)
        people_max = np.argsort(people_count)[::-1]
        PEOPLE_COUNT = 3
        PEOPLE_SELECT = 3
        selected_people = 0
        for id in people_max:
            if people_count[id] > PEOPLE_COUNT and selected_people < PEOPLE_SELECT:
                print(topic_dict['people'][id], people_count[id])
                topic_list[seg_index]['people'].append(topic_dict['people'][id])
                selected_people += 1
            else:
                break
        for j in range(PEOPLE_SELECT-selected_people):
            topic_list[seg_index]['people'].append(None)

        # topic: location
        print("\n### topic: location ###")
        topic_list[seg_index]['location'] = []
        loc_count = []
        for loc in topic_dict['location']:
            loc_count.append(text_seg[i].count(loc.lower()))
        loc_max = np.argsort(loc_count)[::-1]
        LOC_COUNT = 2
        LOC_SELECT = 2
        selected_loc = 0
        for id in loc_max:
            if loc_count[id] > LOC_COUNT and selected_loc < LOC_SELECT:
                print(topic_dict['location'][id], loc_count[id])
                topic_list[seg_index]['location'].append(topic_dict['location'][id])
            else:
                break
        for j in range(LOC_SELECT-selected_loc):
            topic_list[seg_index]['location'].append(None)

        # topic: location
        print("\n### topic: organization ###")
        topic_list[seg_index]['organization'] = []
        org_count = []
        for org in topic_dict['organization']:
            org_count.append(text_seg[i].count(org.lower()))
        org_max = np.argsort(org_count)[::-1]
        ORG_COUNT = 2
        ORG_SELECT = 2
        selected_org = 0
        for id in org_max:
            if org_count[id] > ORG_COUNT and selected_org < ORG_SELECT:
                print(topic_dict['organization'][id], org_count[id])
                topic_list[seg_index]['organization'].append(topic_dict['organization'][id])
            else:
                break
        for j in range(ORG_SELECT-selected_org):
            topic_list[seg_index]['organization'].append(None)

        print("======================================================================")  
    return topic_list
    
def assign_topic(tex_seg, text_seg_words, topic_dict, keywords, w2v_model, SEGTIME):
    num_seg = len(text_seg_words)
    num_key = len(keywords)
    topic_list = {}
    for i in range(num_seg):
        # remove commercial seg
        if len(text_seg_words[i]) < SEGTIME/60*10:
            continue

        seg_index = (i*SEGTIME, (i+1)*SEGTIME)
        topic_list[seg_index] = {}
        # topic: subject
        topic_list[seg_index]['subject'] = []
        sim_matrix = np.zeros(num_key)
        for j in range(num_key):
            avg_sim_key = []
            for key in keywords[j]:
                if text_seg_words[i] != []:
                    sim = [w2v_model.wv.similarity(word, key) for word in text_seg_words[i]]
                    avg_sim_key.append(np.average(sim))
                else:
                    avg_sim_key.append(0)
            sim_matrix[j] = np.average(avg_sim_key)

        topic_id = sim_matrix.argsort()[::-1]
        for j in range(5):
            topic_list[seg_index]['subject'].append(keywords[topic_id[j]])

        # topic: people 
        LAST_SPECIAL = {'donald trump', 'hillary clinton', 'barack obama'}
        topic_list[seg_index]['people'] = []
        people_count = []
        for people in topic_dict['people']:
            names = people.split(',')
            if len(names) == 1:
                cnt = text_seg[i].count(names[0].lower())
            else:
                lastname = names[0].lower()
                firstname = names[1].lower()
                last_cnt = count_name(text_seg[i], lastname)
                full_cnt = count_name(text_seg[i], firstname+' '+lastname)
                if full_cnt == 0:
                    if firstname+' '+lastname in LAST_SPECIAL:
                        cnt = last_cnt
                    else:
                        cnt = 0
                else:
                    cnt = last_cnt + full_cnt
            people_count.append(cnt)
        people_max = np.argsort(people_count)[::-1]
        PEOPLE_COUNT = 3
        PEOPLE_SELECT = 3
        selected_people = 0
        for id in people_max:
            if people_count[id] > PEOPLE_COUNT and selected_people < PEOPLE_SELECT:
                topic_list[seg_index]['people'].append(topic_dict['people'][id])
                selected_people += 1
            else:
                break
        for j in range(PEOPLE_SELECT-selected_people):
            topic_list[seg_index]['people'].append(None)

        # topic: location
        topic_list[seg_index]['location'] = []
        loc_count = []
        for loc in topic_dict['location']:
            loc_count.append(text_seg[i].count(loc.lower()))
        loc_max = np.argsort(loc_count)[::-1]
        LOC_COUNT = 2
        LOC_SELECT = 2
        selected_loc = 0
        for id in loc_max:
            if loc_count[id] > LOC_COUNT and selected_loc < LOC_SELECT:
                topic_list[seg_index]['location'].append(topic_dict['location'][id])
            else:
                break
        for j in range(LOC_SELECT-selected_loc):
            topic_list[seg_index]['location'].append(None)

        # topic: location
        topic_list[seg_index]['organization'] = []
        org_count = []
        for org in topic_dict['organization']:
            org_count.append(text_seg[i].count(org.lower()))
        org_max = np.argsort(org_count)[::-1]
        ORG_COUNT = 2
        ORG_SELECT = 2
        selected_org = 0
        for id in org_max:
            if org_count[id] > ORG_COUNT and selected_org < ORG_SELECT:
                topic_list[seg_index]['organization'].append(topic_dict['organization'][id])
            else:
                break
        for j in range(ORG_SELECT-selected_org):
            topic_list[seg_index]['organization'].append(None)

    return topic_list    

def solve_single_video(video_name, topic_dict, keywords, w2v_model, SEGTIME, show_detail=True):
    # load transcript
    srt_path = '../data/transcripts/' + video_name + '.cc5.srt'
    srt_file = Path(srt_path)
    if not srt_file.is_file():
        srt_path = srt_path.replace('cc5', 'cc1')
        srt_file = Path(srt_path)
        if not srt_file.is_file():
            srt_path = srt_path.replace('cc1', 'align')
            srt_file = Path(srt_path)
            if not srt_file.is_file():
                print("%s does not exist!!!" % srt_path)
                return None
    
    # check uft-8
#     try:
#         file = codecs.open(srt_path, encoding='utf-8', errors='strict')
#         for line in file:
#             pass
#     except UnicodeDecodeError:
#         print("Transcript not encoded in utf-8!!!")
#          return None    
    
    text_seg, text_seg_words = load_transcript(srt_path, w2v_model, SEGTIME)
    if show_detail:
        topic_list = assign_topic_detail(text_seg, text_seg_words, topic_dict, keywords, w2v_model, SEGTIME)
    else:
        topic_list = assign_topic(tex_seg, text_seg_words, topic_dict, keywords, SEGTIME)
    return topic_list

def test_single_video(video_name, topic_dict, keywords, w2v_model):
    
    topic_list = solve_single_video(video_name, topic_dict, keywords, w2v_model, 180, show_detail=True)
    return topic_list

def assign_topic_t(video_list, topic_dict_path, w2v_model, thread_id):
    print("Thread %d start computing..." % (thread_id))
    topic_dict_res = {}
    
    topic_dict, keywords = load_topic_from_dict(w2v_model)
    
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        topic_list = solve_single_video(video_name, topic_dict, keywords, w2v_model, 180)
        
        if topic_list is None:
            continue
            
        topic_dict_res[video_name] = topic_list
        if i % 100 == 0:
            pickle.dump(topic_dict_res, open(topic_dict_path, "wb" ))
    pickle.dump(topic_dict_res, open(topic_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))

def assign_topic_multithread(video_list_path, topic_dict_path, nthread=16):
    video_list = open(video_list_path).read().split('\n')
    del video_list[-1]
    
    # remove exist videos:
    dict_file = Path(topic_dict_path)
    if dict_file.is_file():
        topic_dict = pickle.load(open(topic_dict_path, "rb" ))
        video_list = [video for video in video_list if not video in topic_dict]
    else:
        topic_dict = {}
    
    num_video = len(video_list)
    print(num_video)
    if num_video <= nthread:
        nthread = num_video
        num_video_t = 1
    else:
        num_video_t = math.ceil(1. * num_video / nthread)
    print(num_video_t)
    
    topic_dict_list = []
    for i in range(nthread):
        topic_dict_list.append('../tmp/topic_dict_' + str(i) + '.pkl')
    
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        t = threading.Thread(target=assign_topic_t, args=(video_list_t, topic_dict_list[i], w2v_model, i,))
        t.setDaemon(True)
        thread_list.append(t)
    
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    
    for path in topic_dict_list:
        dict_file = Path(path)
        if not dict_file.is_file():
            continue
        topic_dict_tmp = pickle.load(open(path, "rb" ))
        topic_dict = {**topic_dict, **topic_dict_tmp}
        
    pickle.dump(topic_dict, open("../data/topic_dict.pkl", "wb" ))    
    
    