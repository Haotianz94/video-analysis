import numpy as np
import pysrt
import gensim
import textacy
from textblob import TextBlob
import pickle
import time
from pathlib import Path
import codecs
import math
import multiprocessing as mp
# import xml.etree.ElementTree as ET
from utility import *

def get_transcript_seg(start, end, transcript):
    TRANSCRIPT_DELAY = 6
    SEGTIME = 180
    MIN_THRESH = 30
    replace_list = ['\n', '>', '.', ',', '?', '!', '-', '(', ')', '[', ']']
    text_seg = []
    
    if len(transcript) == 0:
        return [], 0
    # skip commercial block
    for i in range(len(transcript)):
        t = transcript[i]
        if get_second(t[2]) - TRANSCRIPT_DELAY <= start:
            continue
        else:
            break
    transcript = transcript[i:]
        
    if end == None:
        seg_start = start
        seg_end = start + SEGTIME
        text = ''
        for t in transcript:
            if get_second(t[1]) - TRANSCRIPT_DELAY < seg_end:
                text += t[0] + ' '
            else:
#                 print(seg_end)
                for token in replace_list:
                    text = text.replace(token, ' ')
                text = text.lower()
                text_seg.append({'text':text, 'seg':(seg_start, seg_end)})
                text = ''
                seg_start = seg_end
                seg_end += SEGTIME
        if text != '':
            for token in replace_list:
                text = text.replace(token, ' ')
            text = text.lower()
            text_seg.append({'text':text, 'seg':(seg_start, get_second(t[2]))})
        return text_seg, len(transcript)        
    
    seg_start = start
    if start + SEGTIME + MIN_THRESH < end:
        seg_end = start + SEGTIME
    else:
        seg_end = end
    text = ''
    for i in range(len(transcript)):
        t = transcript[i]
        if get_second(t[1]) - TRANSCRIPT_DELAY < seg_end:
            text += t[0] + ' '
        else:
#             print(seg_end)
            for token in replace_list:
                text = text.replace(token, ' ')
            text = text.lower()
            text_seg.append({'text':text, 'seg':(seg_start, seg_end)})
            text = ''
            if seg_end == end:
                return text_seg, i
            seg_start = seg_end
            if seg_end + SEGTIME + MIN_THRESH < end:
                seg_end += SEGTIME
            else:
                seg_end = end
    if text != '':      
        for token in replace_list:
            text = text.replace(token, ' ')
        text = text.lower()
        text_seg.append({'text':text, 'seg':(seg_start, get_second(t[2]) )})
    return text_seg, len(transcript)        

def load_transcript(srt_path, w2v_model, com_list):    
    # Load transcripts
    subs = pysrt.open(srt_path)
    transcript = []
    for sub in subs:
        transcript.append((sub.text, tuple(sub.start)[:3], tuple(sub.end)[:3]))
    
    # split transcript into segment
    text_seg = []
    text = ''
    lc = len(com_list)
    if lc == 0:
        text_seg, i = get_transcript_seg(0, None, transcript)
    else:
        for c in range(lc):
            if c == 0:
                seg, i = get_transcript_seg(0, get_second(com_list[0][0][1]), transcript)
            else:
                seg, i =get_transcript_seg(get_second(com_list[c-1][1][1]),get_second(com_list[c][0][1]), transcript)
            text_seg += seg
            transcript = transcript[i:]
        seg, i = get_transcript_seg(get_second(com_list[-1][1][1]), None, transcript)
        text_seg += seg
    
    # Extract PROPN and NOUN using textacy
    docs = [textacy.doc.Doc(t['text'], lang=u'en') for t in text_seg]
    corpus = textacy.Corpus(u'en', docs=docs)

    for i in range(len(text_seg)):
        words = list(textacy.extract.words(corpus.docs[i], filter_nums=True, include_pos=['PROPN', 'NOUN']))
        words = [str(word) for word in words if len(str(word)) > 1 and str(word) in w2v_model.vocab]
        text_seg[i]['words'] = words
    
    
#     text_seg_words = [list(textacy.extract.words(doc, filter_nums=True, include_pos=['PROPN', 'NOUN'])) for doc in corpus.docs]
#     # remove word not in w2v model
#     text_seg_words = [ [str(word) for word in seg if len(str(word)) > 1 and str(word) in w2v_model.vocab] for seg in text_seg_words]
    return text_seg

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
            if start+ln < len(text) and text[start+ln] != ':' and not text[start+ln].isalpha():
                cnt += 1
            elif start+ln == len(text):
                cnt += 1
    return cnt

def assign_topic(text_seg, topic_dict, w2v_model, show_detail=False):
    num_seg = len(text_seg)
#     num_key = len(topic_dict['subject'])
    topic_list = {}
    for seg in text_seg:
        # remove commercial seg
        if len(seg['words']) < 30:
            continue

        seg_index = seg['seg']
        topic_list[seg_index] = {}
        # topic: subject
        if show_detail:
            print('\n',seg_index)
            print("\n### subject ###")
        topic_list[seg_index]['subject'] = []
        # by similarity
        similiraty = []
        for sub in topic_dict['subject']:
            sim = [w2v_model.wv.similarity(word, sub) for word in seg['words']]
            similiraty.append(np.average(sim))

        subject_id = np.argsort(similiraty)[::-1]
        for j in range(5):
            if show_detail:
                print(similiraty[subject_id[j]], topic_dict['subject'][subject_id[j]])
            topic_list[seg_index]['subject'].append(topic_dict['subject'][subject_id[j]])
        if show_detail:
            print("\nsubject related words:")
            key = topic_dict['subject'][subject_id[0]]
            sim = np.array([w2v_model.wv.similarity(word, key) for word in seg['words']])
            sim_word_id = sim.argsort()[::-1]
            sim_words = [seg['words'][sim_word_id[j]] for j in range(20) if j < len(seg['words'])]
            print(sim_words)
            
        # by exact match
#         subject_count = []
#         for subject in topic_dict['subject']:
#             subject_count.append(seg['text'].count(subject.lower()))
#         subject_max = np.argsort(subject_count)[::-1]
#         SUBJECT_COUNT = 1
#         for id in subject_max:
#             if subject_count[id] > SUBJECT_COUNT:
#                 if show_detail:
#                     print(topic_dict['subject'][id], subject_count[id])
#                 topic_list[seg_index]['subject'].append(topic_dict['subject'][id])
#             else:
#                 break
        
        # topic: phrase
        if show_detail:
            print("\n### phrase ###")
        topic_list[seg_index]['phrase'] = []
        phrase_count = []
        for phrase in topic_dict['phrase']:
            phrase_count.append(seg['text'].count(phrase.lower()))
        phrase_max = np.argsort(phrase_count)[::-1]
        PHRASE_COUNT = 1
        for id in phrase_max:
            if phrase_count[id] > PHRASE_COUNT:
                if show_detail:
                    print(topic_dict['phrase'][id], phrase_count[id])
                topic_list[seg_index]['phrase'].append(topic_dict['phrase'][id])
            else:
                break
        
        # topic: people 
        LAST_SPECIAL = {'donald trump', 'hillary clinton', 'barack obama'}
        if show_detail:
            print("\n### people ###")
        topic_list[seg_index]['people'] = []
        people_count = []
        for people in topic_dict['people']:
            names = people.split(',')
            if len(names) == 1:
                cnt = seg['text'].count(names[0].lower())
            else:
                lastname = names[0].lower()
                firstname = names[1].lower()
                last_cnt = count_name(seg['text'], lastname)
                full_cnt = count_name(seg['text'], firstname+' '+lastname)
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
        for id in people_max:
            if people_count[id] > PEOPLE_COUNT:
                if show_detail:
                    print(topic_dict['people'][id], people_count[id])
                topic_list[seg_index]['people'].append(topic_dict['people'][id])
            else:
                break

        # topic: location
        if show_detail:
            print("\n### location ###")
        topic_list[seg_index]['location'] = []
        loc_count = []
        for loc in topic_dict['location']:
            loc_count.append(seg['text'].count(loc.lower()))
        loc_max = np.argsort(loc_count)[::-1]
        LOC_COUNT = 2
        for id in loc_max:
            if loc_count[id] > LOC_COUNT:
                if show_detail:
                    print(topic_dict['location'][id], loc_count[id])
                topic_list[seg_index]['location'].append(topic_dict['location'][id])
            else:
                break

        # topic: location
        if show_detail:
            print("\n### organization ###")
        topic_list[seg_index]['organization'] = []
        org_count = []
        for org in topic_dict['organization']:
            org_count.append(seg['text'].count(org.lower()))
        org_max = np.argsort(org_count)[::-1]
        ORG_COUNT = 2
        for id in org_max:
            if org_count[id] > ORG_COUNT:
                if show_detail:
                    print(topic_dict['organization'][id], org_count[id])
                topic_list[seg_index]['organization'].append(topic_dict['organization'][id])
            else:
                break
        
        # sentiment analysis
        if show_detail:
            print("\n### sentiment ###")
        blob = TextBlob(seg['text'])
        sen = blob.sentences[0].sentiment
        if show_detail:
            print(sen)
        topic_list[seg_index]['sentiment'] = (sen.polarity, sen.subjectivity)
        
        # store text
        topic_list[seg_index]['transcript'] = seg['text']
        
    return topic_list
    
def solve_single_video(video_name, topic_dict, com_list, w2v_model, show_detail=True):
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
    
    text_seg = load_transcript(srt_path, w2v_model, com_list)
    topic_list = assign_topic(text_seg, topic_dict, w2v_model, show_detail)
    return topic_list

def test_single_video(video_name, topic_dict, com_dict, w2v_model):
    
    topic_list = solve_single_video(video_name, topic_dict, com_dict[video_name], w2v_model, show_detail=True)
    return topic_list

def assign_topic_t(video_list, topic_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    topic_dict_res = {}
    
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    topic_dict = pickle.load(open('../data/topic_dict.pkl', 'rb'))
    com_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        if not video_name in com_dict:
            continue
        topic_list = solve_single_video(video_name, topic_dict, com_dict[video_name], w2v_model, False)
        
        if topic_list is None:
            continue
            
        topic_dict_res[video_name] = topic_list
        if i % 20 == 0:
            pickle.dump(topic_dict_res, open(topic_dict_path, "wb" ))
    pickle.dump(topic_dict_res, open(topic_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))
    
def assign_topic_multiprocess(video_list_path, topic_dict_path, nprocess=16):
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
    if num_video <= nprocess:
        nprocess = num_video
        num_video_t = 1
    else:
        num_video_t = math.ceil(1. * num_video / nprocess)
    print(num_video_t)
    
    topic_dict_list = []
    for i in range(nprocess):
        topic_dict_list.append('../tmp/topic_dict_' + str(i) + '.pkl')
    
#     w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
#     topic_dict, keywords = load_topic_from_dict(w2v_model)
    ctx = mp.get_context('spawn')
    process_list = []
    for i in range(nprocess):
        if i != nprocess - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        p = ctx.Process(target=assign_topic_t, args=(video_list_t, topic_dict_list[i], i,))
        process_list.append(p)
    
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
    
    for path in topic_dict_list:
        dict_file = Path(path)
        if not dict_file.is_file():
            continue
        topic_dict_tmp = pickle.load(open(path, "rb" ))
        topic_dict = {**topic_dict, **topic_dict_tmp}
        
    pickle.dump(topic_dict, open(topic_dict_path, "wb" ))
    
# def sentiment_analysis():
    
    