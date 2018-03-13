def solve_thread(video_list, tmp_dict_path, thread_id):
    print("Thread %d start computing..." % (thread_id))
    
    ## load dict
#     face_dict = pickle.load(open('../data/face_dict.pkl', 'rb'))
#     com_dict = pickle.load(open('../data/commercial_dict.pkl', 'rb'))
    res_dict = {}
    
    for i in range(len(video_list)):
        video_name = video_list[i]
        print("Thread %d start %dth video: %s" % (thread_id, i, video_name))
        res = solve_single_video(video_name)

        res_dict[video_name] = res
        if i % 100 == 0:
            pickle.dump(res_dict, open(tmp_dict_path, "wb" ))
            
        pickle.dump(res_dict, open(tmp_dict_path, "wb" ))
    print("Thread %d finished computing..." % (thread_id))

def solve_parallel(video_list_path, res_dict_path=None, nthread=16, use_process=True):
    video_list = open(video_list_path).read().split('\n')
    
    dict_file = Path(res_dict_path)
    if dict_file.is_file():
        res_dict = pickle.load(open(res_dict_path, "rb" ))
        video_list = [video for video in video_list if video not in res_dict]
    else:
        res_dict = {}

    num_video = len(video_list)
    print(num_video)
    if num_video == 0:
        return 
    if num_video <= nthread:
        nthread = num_video
        num_video_t = 1
    else:
        num_video_t = math.ceil(1. * num_video / nthread)
    print(num_video_t)
    
    tmp_dict_list = []
    for i in range(nthread):
        tmp_dict_list.append('../tmp/anchor_dict_' + str(i) + '.pkl')

    if use_process:
        ctx = mp.get_context('spawn')
    thread_list = []
    for i in range(nthread):
        if i != nthread - 1:
            video_list_t = video_list[i*num_video_t : (i+1)*num_video_t]
        else:
            video_list_t = video_list[i*num_video_t : ]
        if use_process:
            t = ctx.Process(target=solve_thread, args=(video_list_t, tmp_dict_list[i], i,))
        else:
            t = threading.Thread(target=solve_thread, args=(video_list_t, tmp_dict_list[i], i,))
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
        res_dict = {**res_dict, **res_dict_tmp}
    
    pickle.dump(res_dict, open(res_dict_path, "wb" ))  