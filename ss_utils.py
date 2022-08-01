def com_send(mess, PORT, HOST,socket, AF_INET, SOCK_STREAM):
    sock = socket(AF_INET, SOCK_STREAM)
    try:
        # 通信の確立
        sock.connect((HOST, PORT))
        # メッセージ送信
        sock.sendall(mess)
        # 通信の終了
        sock.close()

    except Exception as e:
        print(e)


def com_send_str(mess, PORT, HOST,socket, AF_INET, SOCK_STREAM):

    while True:
        try:
            # 通信の確立
            sock = socket(AF_INET, SOCK_STREAM)
            sock.connect((HOST, PORT))

            # メッセージ送信
            sock.send(mess.encode('utf-8'))

            # 通信の終了
            sock.close()
            break

        except Exception as e:
            print(e)
            pass
            

def com_receive(PORT, HOST,socket, AF_INET, SOCK_STREAM):

    # 通信の確立
    MAX_MESSAGE = 2048
    NUM_THREAD = 4
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(NUM_THREAD)

    # メッセージ受信ループ
    try:
        conn, addr = sock.accept()

    except KeyboardInterrupt:
        print("ctrl c")

    else:
        all_mess = b''

        try:
            while True:
                mess = conn.recv(MAX_MESSAGE)
                if not mess:
                    break

                all_mess += mess

            return all_mess

        except Exception as e:
            print('受信処理エラー発生')
            print(e)




class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m'  # 反転
    ACCENT = '\033[01m'  # 強調
    FLASH = '\033[05m'  # 点滅
    RED_FLASH = '\033[05;41m'  # 赤背景+点滅
    END = '\033[0m'


def get_closest_index(start_timestamp, end_timestamp, data_timestamp_np, np):

    closest_index_start = getNearestValue(
        data_timestamp_np, start_timestamp, "start", 250, np)
    closest_index_end = getNearestValue(
        data_timestamp_np, end_timestamp, "end", 250, np)

    return closest_index_start, closest_index_end


def getNearestValue(np_list, num, POSITION, RANGE, np):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値

    """
    lis = np.where((np_list > (num - RANGE)) & (np_list < (num + RANGE)))[0]
    idx = 0

    if POSITION == "start":
        idx = lis[0]

    elif POSITION == "center":
        idx = lis[(len(lis)) // 2]

    elif POSITION == "end":
        idx = lis[-1]

    return idx


#integrated subsampling
def subsample(start_timestamp, end_timestamp, step, data_timestamp_np, data_frames_np, np):
    start_timestamp = int(1000*start_timestamp)
    end_timestamp = int(1000*end_timestamp)
    
    closest_index_start, closest_index_end = get_closest_index(start_timestamp, end_timestamp, data_timestamp_np, np)
    
    print("start_timestamp",start_timestamp,"end_timestamp", end_timestamp, "closest_index_start", closest_index_start,"closest_index_end", closest_index_end)

    subsampled_frames = []


    tmp_data_timestamp_np = data_timestamp_np[closest_index_start:closest_index_end]
    tmp_data_frames = data_frames_np[closest_index_start:closest_index_end]
    
    print("tmp_data_timestamp_np", len(tmp_data_timestamp_np), "tmp_data_frames",len(tmp_data_frames))
    
    #タイムスタンプ でforループ回す．
    timestamp_count = data_timestamp_np[closest_index_start]
    odd = True
    cache_frame = None
    
    while True:
        #print(timestamp_count)
        #もしtimestamp_countがend_timestampを超えたら止める．
        
        if timestamp_count > data_timestamp_np[closest_index_end]:
            break
        
        else:
            #最も近いindexをtmp_data_timestamp_npから探す．
            tmp_index = getNearestValue(tmp_data_timestamp_np, timestamp_count, "center",50, np)
            
            #print(tmp_data_timestamp_np[tmp_index], timestamp_count)
            
            #print(tmp_data_frames[tmp_index])
            
            if odd == True:
                cache_frame = tmp_data_frames[tmp_index]
                odd = False
            
            else:
                #そのindexを使って，timp_data_framesからsubsampled_framesに入れる．
                subsampled_frames.append(list(tmp_data_frames[tmp_index] + cache_frame))
                odd = True
            
            timestamp_count += step


    return subsampled_frames


def statsMSD(sentence, typed, np):
    a = " " + sentence
    b = " " + typed
    msd = np.zeros((len(a), len(b)))
    mfa = np.zeros((len(a), len(b)))
    minn = 0
    sub = 0
    notInA = 0

    for i in range(len(a)):
        msd[i, 0] = i
        mfa[i, 0] = 0

    for j in range(len(b)):
        msd[0, j] = j
        mfa[0, j] = j

    for i in range(1, len(a)):
        for j in range(1, len(b)):
            minn = msd[i - 1, j] + 1
            notInA = msd[i, j - 1] + 1
            sub = msd[i - 1, j - 1]

            if a[i] != b[j]:
                sub += 1

            mfa[i, j] = mfa[i - 1, j]

            if sub < minn:
                minn = sub
                mfa[i, j] = mfa[i - 1, j - 1]

            if notInA < minn:
                minn = notInA
                mfa[i, j] = mfa[i, j - 1] + 1

            msd[i, j] = minn

    ret = []
    ret.append(msd[len(a) - 1, len(b) - 1])
    ret.append(mfa[len(a) - 1, len(b) - 1])

    return ret


def statsF(keyPresses, np):
    ret = 0
    for i in range(len(keyPresses)):
        temp = keyPresses[i]
        if temp is "<" or temp is "^":
            ret += 1

    return ret


def statsIF(sentence, typed, keyPresses, np):
    return statsF(keyPresses, np) - (len(typed) - len(sentence))


def statsC(sentence, typed, np):
    msd = statsMSD(sentence, typed, np)
    length = min(len(typed), len(sentence))
    return length - (msd[0] - msd[1])


def statsINF(sentence, typed, np):
    return statsMSD(sentence, typed, np)[0]


def statsTotalErrorRate(sentence, typed, keyPresses, np):
    INF = statsINF(sentence, typed, np)
    C = statsC(sentence, typed, np)
    IF = statsIF(sentence, typed, keyPresses, np)
    F = statsF(keyPresses, np)
    TER = (INF + IF) / (C + INF + IF)

    return TER   


def change_state(result_np, state):
    pos_lis = [False, False, False]

    if result_np[:25].sum() > 5:
        pos_lis[0] = True

    if result_np[19:36].sum() > 5:
        pos_lis[1] = True

    if result_np[30:50].sum() > 7:
        pos_lis[2] = True

    if pos_lis == [True, False, False]:
        if state == 0:
            state = 1
        if state == 1:
            state = 1
        if state == 2:
            state = 0

    if pos_lis == [False, True, False]:
        if state == 0:
            state = 0
        if state == 1:
            state = 2
        if state == 2:
            state = 2

    if pos_lis == [False, False, True]:
        if state == 0:
            state = 0
        if state == 1:
            state = 0
        if state == 2:
            state = 3

    return state

def change_2state(result_np, state):
    pos_lis = [False, False]
    #print("state",state)

    if result_np[:50].sum() > 10:
        #print("result_np[:50].sum() > 5:")
        pos_lis[0] = True

    if result_np[40:123].sum() > 10:
        #print("result_np[51:123].sum() > 5")
        pos_lis[1] = True

    #print("pos_lis", pos_lis)

    if pos_lis == [True, False]:
        if state == 0:
            print("changed_state_to_1")
            state = 1

    elif pos_lis == [False, True]:
        if state == 0:
            state = 0
        if state == 1:
            print("changed_state_to_2")
            state = 2

    elif pos_lis == [True, True]:
        print("all zero")
        state = 0


    elif pos_lis == [False, False]:
        if state == 2:
            print("changed_state_to_0")
            state = 0
        if state == 1:
            print("changed_state_to_0")
            state = 0        

    #print("returned state", state)
    return state



def integrate_with_lm_8(result, previous_word, df, np):
    result_lines = result.split('\n')

    recognized_words = []
    recognized_prob = []
    recognized_words.append(result_lines[3].split(' ')[0])
    recognized_words.append(result_lines[7].split(' ')[0])
    recognized_words.append(result_lines[11].split(' ')[0])
    recognized_words.append(result_lines[15].split(' ')[0])
    recognized_words.append(result_lines[19].split(' ')[0])
    recognized_words.append(result_lines[23].split(' ')[0])
    recognized_words.append(result_lines[27].split(' ')[0])
    recognized_words.append(result_lines[31].split(' ')[0])
    recognized_words.append(result_lines[35].split(' ')[0])
    recognized_words.append(result_lines[39].split(' ')[0])
    recognized_words.append(result_lines[43].split(' ')[0])
    recognized_words.append(result_lines[47].split(' ')[0])
    recognized_words.append(result_lines[51].split(' ')[0])
    recognized_words.append(result_lines[55].split(' ')[0])
    recognized_words.append(result_lines[59].split(' ')[0])
    recognized_words.append(result_lines[63].split(' ')[0])
    recognized_words = np.array(recognized_words)

    recognized_prob.append(10**float(result_lines[3].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[7].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[11].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[15].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[19].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[23].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[27].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[31].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[35].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[39].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[43].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[47].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[51].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[55].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[59].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[63].split(' ')[1]))

    recognized_prob = np.array(recognized_prob)
    regularized_recognized_prob = recognized_prob / recognized_prob.sum()

    try:
        current_prob = []
        for current_word in recognized_words:
            transition_prob = df.at[previous_word, current_word]

            current_prob.append(transition_prob)

        current_prob = np.array(current_prob)
        regularized_current_prob = current_prob / current_prob.sum()

        return recognized_words[np.argsort(regularized_recognized_prob * regularized_current_prob)[::-1]][:5]

    except Exception as E:
        print(E)
        return recognized_word[np.argsort(regularized_recognized_prob)[::-1]][:5]

def integrate_with_lm_10(result, previous_word, df, np):
    result_lines = result.split('\n')

    recognized_words = []
    recognized_prob = []

    recognized_words.append(result_lines[3].split(' ')[0])
    recognized_words.append(result_lines[7].split(' ')[0])
    recognized_words.append(result_lines[11].split(' ')[0])
    recognized_words.append(result_lines[15].split(' ')[0])
    recognized_words.append(result_lines[19].split(' ')[0])
    recognized_words.append(result_lines[23].split(' ')[0])
    recognized_words.append(result_lines[27].split(' ')[0])
    recognized_words.append(result_lines[31].split(' ')[0])
    recognized_words.append(result_lines[35].split(' ')[0])
    recognized_words.append(result_lines[39].split(' ')[0])

    recognized_words = np.array(recognized_words)

    recognized_prob.append(10**float(result_lines[3].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[7].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[11].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[15].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[19].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[23].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[27].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[31].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[35].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[39].split(' ')[1]))

    recognized_prob = np.array(recognized_prob)
    regularized_recognized_prob = recognized_prob / recognized_prob.sum()

    try:
        current_prob = []
        #current_probとはなんだろう．．．transition_probを追加しているが．．．
        for current_word in recognized_words:
            #previous wordとcurrent wordを入れると，transition probが出力されるが，これは遅くなる要因になりうるので，いずれnumpy行列に変えたい．
            transition_prob = df.at[previous_word, current_word]

            current_prob.append(transition_prob)

        current_prob = np.array(current_prob)
        regularized_current_prob = current_prob / current_prob.sum()

        # 確率が高い方から上位6つ？を返す．
        return recognized_words[np.argsort(regularized_recognized_prob * regularized_current_prob)[::-1]][:5]

    except Exception as E:
        print(E)
        return recognized_word[np.argsort(regularized_recognized_prob)[::-1]][:5]

def integrate_with_lm_5(result, previous_word, df, np):
    result_lines = result.split('\n')

    recognized_words = []
    recognized_prob = []

    recognized_words.append(result_lines[3].split(' ')[0])
    recognized_words.append(result_lines[7].split(' ')[0])
    recognized_words.append(result_lines[11].split(' ')[0])
    recognized_words.append(result_lines[15].split(' ')[0])
    recognized_words.append(result_lines[19].split(' ')[0])

    recognized_words = np.array(recognized_words)

    recognized_prob.append(10**float(result_lines[3].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[7].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[11].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[15].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[19].split(' ')[1]))

    recognized_prob = np.array(recognized_prob)
    regularized_recognized_prob = recognized_prob / recognized_prob.sum()

    try:
        current_prob = []
        #current_probとはなんだろう．．．transition_probを追加しているが．．．
        for current_word in recognized_words:
            #previous wordとcurrent wordを入れると，transition probが出力されるが，これは遅くなる要因になりうるので，いずれnumpy行列に変えたい．
            transition_prob = df.at[previous_word, current_word]

            current_prob.append(transition_prob)

        current_prob = np.array(current_prob)
        regularized_current_prob = current_prob / current_prob.sum()

        # 確率が高い方から上位6つ？を返す．
        return recognized_words[np.argsort(regularized_recognized_prob * regularized_current_prob)[::-1]][:4]

    except Exception as E:
        print(E)
        return recognized_word[np.argsort(regularized_recognized_prob)[::-1]][:5]


def integrate_with_lm_5_with_length_penalty(result, previous_word, df, length_penalty, np):
    result_lines = result.split('\n')

    recognized_words = []
    recognized_prob = []

    recognized_words.append(result_lines[3].split(' ')[0])
    recognized_words.append(result_lines[7].split(' ')[0])
    recognized_words.append(result_lines[11].split(' ')[0])
    recognized_words.append(result_lines[15].split(' ')[0])
    recognized_words.append(result_lines[19].split(' ')[0])

    recognized_words = np.array(recognized_words)

    recognized_prob.append(10**float(result_lines[3].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[7].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[11].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[15].split(' ')[1]))
    recognized_prob.append(10**float(result_lines[19].split(' ')[1]))

    recognized_prob = np.array(recognized_prob)
    regularized_recognized_prob = recognized_prob / recognized_prob.sum()


    #length penalty prob
    length_prob = []

    length_prob.append(1 - length_penalty * len(result_lines[3].split(' ')[0]))
    length_prob.append(1 - length_penalty * len(result_lines[7].split(' ')[0]))
    length_prob.append(1 - length_penalty * len(result_lines[11].split(' ')[0]))
    length_prob.append(1 - length_penalty * len(result_lines[15].split(' ')[0]))
    length_prob.append(1 - length_penalty * len(result_lines[19].split(' ')[0]))

    length_prob = np.array(length_prob)
    regularized_length_prob = length_prob / length_prob.sum()




    try:
        current_prob = []
        #current_probとはなんだろう．．．transition_probを追加しているが．．．
        for current_word in recognized_words:
            #previous wordとcurrent wordを入れると，transition probが出力されるが，これは遅くなる要因になりうるので，いずれnumpy行列に変えたい．
            transition_prob = df.at[previous_word, current_word]

            current_prob.append(transition_prob)

        current_prob = np.array(current_prob)
        regularized_current_prob = current_prob / current_prob.sum()

        # 確率が高い方から上位6つ？を返す．
        print(recognized_words, regularized_recognized_prob, regularized_current_prob, regularized_length_prob, regularized_recognized_prob * regularized_current_prob * regularized_length_prob)
        return recognized_words[np.argsort(regularized_recognized_prob * regularized_current_prob * regularized_length_prob)[::-1]][:4]

    except Exception as E:
        print(E)
        return recognized_word[np.argsort(regularized_recognized_prob)[::-1]][:5]




def make_log_dir(log_dir_thistime, length_of_filelist, os):
    try:
        os.mkdir(log_dir_thistime)
        os.mkdir(log_dir_thistime + "frames")
        os.mkdir(log_dir_thistime + "timestamps")

    except BaseException:
        pass