# インポート系
from smart_palate import *
import time
import threading
#import csv
import random
import pickle
import numpy as np
from collections import deque
import re
import pandas as pd
from pynput import keyboard
import curses
import multiprocessing
import os
import subprocess
import matplotlib.pyplot as plt
import ss_utils
import sys


# socket
from socket import socket, AF_INET, SOCK_STREAM, gethostname, gethostbyname


interface_to_reader_PORT = 50010
sensitivity = 40

interface_to_reader_PORT = 50010

vis_HOST = "localhost"
step = 10
CHR_CAN = '\18'
CHR_EOT = '\04'
MAX_MESSAGE = 2048
NUM_THREAD = 4

#corpus setting
dir = "dataset/"
num = 1


#pycolor setting
class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'




# new version
def get_closest_index(start_timestamp, end_timestamp, data_timestamp_np):

    closest_index_start = getNearestValue(
        data_timestamp_np, start_timestamp, "start", 250)
    closest_index_end = getNearestValue(
        data_timestamp_np, end_timestamp, "end", 100)

    return closest_index_start, closest_index_end


def getNearestValue(np_list, num, POSITION, RANGE):
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


def subsample(
        start_timestamp,
        end_timestamp,
        step,
        data_timestamp_np,
        data_frames_np):
    start_timestamp = int(1000 * start_timestamp)
    end_timestamp = int(1000 * end_timestamp)

    closest_index_start, closest_index_end = get_closest_index(
        start_timestamp, end_timestamp, data_timestamp_np)

    subsampled_frames = []

    tmp_data_timestamp_np = data_timestamp_np[closest_index_start:closest_index_end]
    tmp_data_frames = data_frames_np[closest_index_start:closest_index_end]

    # タイムスタンプ でforループ回す．
    timestamp_count = data_timestamp_np[closest_index_start]
    while True:
        # もしtimestamp_countがend_timestampを超えたら止める．

        if timestamp_count > data_timestamp_np[closest_index_end]:
            break

        else:
            # 最も近いindexをtmp_data_timestamp_npから探す．
            tmp_index = getNearestValue(
                tmp_data_timestamp_np, timestamp_count, "center", 20)

            # そのindexを使って，timp_data_framesからsubsampled_framesに入れる．
            subsampled_frames.append(list(tmp_data_frames[tmp_index]))

            timestamp_count += step

    return subsampled_frames




#　センサからひたすらデータを収集し続けるプロセス．　取得したフレームとタイムスタンプ(time.time())を一つのリストにして，reader_to_writer_q（キュー）に追加し続ける．
#　openCVでカメラから取得するなら，以下のような感じで良いと思う．
"""
def reader_multi(reader_to_writer_q, reader_flag):
    print("reader started.")
    smart_palate = SmartPalate()
    sleep = False
    
    while True:
        result = None
        try:
            ret, frame = cap.read()
        except:
            print(sys.exc_info()[0])
            sleep = True
            pass

        if ret:
            reader_to_writer_q.put([frame, time.time()])
            reader_flag.set()
        elif sleep:
            print('sleep!')
            sleep = False
            time.sleep(0.003)
"""
def reader_multi(reader_to_writer_q, reader_flag):
    print("reader started.")
    smart_palate = SmartPalate()
    sleep = False

    while True:
        result = None
        try:
            result = smart_palate.readFrame()
        except:
            print(sys.exc_info()[0])
            print('USB not stable! If this message appears a log, consider switching USB port or Cable!')
            # TODO: Naoki, if this exception occurs when the user is recording a phrase, make the user redo the phrase
            sleep = True
            pass

        if result is not None:
            # sleep = True
            # print(len(result[1]))
            frame = result[1]
            reader_to_writer_q.put([frame, time.time()])
            reader_flag.set()

            if reader_to_writer_q.full():
                print(ss_utils.pycolor.BLUE + "reader_being_blocked." + ss_utils.pycolor.END)


        elif sleep:
            print('sleep!')
            sleep = False
            time.sleep(0.003)


# 20220801　だいぶ昔に書いたのでflagとかキューの役割がわからなくなってるが，
# 常に
#            result = reader_to_writer_q.get()
#            d.append(result)
#でfor loopが回ってて，dというデキューに追加し続ける．
# if not starttimestamp_q.empty() and not endtimestamp_q.empty(): メインのプロセスから収録開始・終了を示すタイムスタンプが飛んできたら，デキューの中から該当のフレームを取得して保存する．
基本的にreader_to_writer_qからフレームを読み出して，
def writer_multi(
        reader_to_writer_q,
        vis_PORT,
        vis_HOST,
        starttimestamp_q,
        endtimestamp_q,
        delete_flag,
        detect_flag,
        reader_flag,
        activity_flag,
        writer_to_main_q
        ):

    print("writer started.")
    d = deque(maxlen=1000)
    delete_deque = deque(maxlen=sensitivity)
    state = 0
    #fig = plt.figure()
    plt.show()
    electrodes = 100
    vis_counter = 2

    #step = 10

    while True:
        # データ保存処理
        if not starttimestamp_q.empty() and not endtimestamp_q.empty():
            end_timestamp = endtimestamp_q.get()
            start_timestamp = starttimestamp_q.get()

            data = list(d)
            # 扱いやすい形に変える
            data_frames_np = np.array([[float(y) for y in x]
                                       for x in [row[0] for row in data]])
            #data_frames_np = np.array(d)[:, 0]
            # print(data_frames_np)
            data_timestamps = [int(1000 * row[1]) for row in data]
            data_timestamp_np = np.array(data_timestamps)
            #data_timestamp_np = np.array(d)[:, 2]
            
            try:
                subsampled_frames = subsample(
                    start_timestamp,
                    end_timestamp,
                    step,
                    data_timestamp_np,
                    data_frames_np,
                    )
            except:
                subsampled_frames = [[0.0] * 16] * 50


            #メインプロセスに通知する．
            writer_to_main_q.put(subsampled_frames)

            d.clear()

        #キャッチングアップ
        elif reader_to_writer_q.empty() is not True:
            result = reader_to_writer_q.get()
            d.append(result)


        elif reader_to_writer_q.empty() is True:
            reader_flag.wait()
            result = reader_to_writer_q.get()
            reader_flag.clear()

            if vis_counter ==0:
                mesg = ' '.join(map(str, result[0]))
                ss_utils.com_send_str(mesg, vis_PORT, vis_HOST,socket, AF_INET, SOCK_STREAM)
                vis_counter = 3

            else:
                vis_counter -= 1

            d.append(result)

def on_press(key):
    if key == keyboard.Key.cmd_r:
        print(pycolor.RED + "recording..." + pycolor.END)
        start_flag.set()
        activity_flag.set()

    elif key == keyboard.Key.shift_r:

        goback_flag.set()

def on_release(key):
    if key == keyboard.Key.cmd_r:
        end_flag.set()
        time.sleep(0.2)







if __name__ == '__main__':
    vis_PORT = random.randint(49200, 60000)

    multiprocessing.set_start_method('spawn')

    np.seterr(divide='ignore', invalid='ignore')


    if not os.path.exists(dir):
        os.mkdir(dir)

    if os.path.isfile(dir + "word_list.pkl"):
        with open(dir + 'word_list.pkl', 'rb') as f:
            transcripts = pickle.load(f)

        print("transcripts",transcripts)

    else:
        with open(dir + 'word_list.pkl', 'rb') as f:
            transcripts = pickle.load(f)

        print("transcripts",transcripts)


    if os.path.isfile(dir + "word_index.pkl"):

        with open(dir + 'word_index.pkl', 'rb') as f:
            word_index = pickle.load(f)

        print("word_index",word_index)

    else:

        word_index = 0
        with open(dir + 'word_index.pkl', 'wb') as f:
            pickle.dump(word_index, f)


    print(len(transcripts))




    starttimestamp_q = multiprocessing.Queue(maxsize=1)
    endtimestamp_q = multiprocessing.Queue(maxsize=1)


    start_flag = threading.Event()
    end_flag = threading.Event()

    # start

    cmd = "python3 visualizer.py " + str(vis_PORT)
    print(cmd)
    #stdout=subprocess.PIPE, stderr=subprocess.PIPE このライナーでエラーが表示されなくなる．．．ある意味危険．
    vis_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #print("process id = %s" % proc.pid)
    # 読み上げるスクリプトの読み込み
    time.sleep(2)

    #f = open("phrases3.txt")
    #data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    #f.close()
    #lines1 = re.split('\n', data1)  # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
    #random.shuffle(lines1)
    # print(lines1)
    #len(lines1)

    timer_flag = threading.Event()
    finish_e = threading.Event()
    input_flag = threading.Event()
    delete_flag = multiprocessing.Event()
    enter_flag = threading.Event()
    reader_flag = multiprocessing.Event()
    activity_flag = multiprocessing.Event()

    writer_to_main_flag = multiprocessing.Event()

    writer_to_main_q = multiprocessing.Queue()

    reader_to_writer_q = multiprocessing.Queue()

    detect_flag = multiprocessing.Event()

    read_to_detect_q = multiprocessing.Queue()



    reader = multiprocessing.Process(
        target=reader_multi, args=(
            reader_to_writer_q, reader_flag))
    reader.daemon = True
    reader.start()


    writer_process = multiprocessing.Process(
        target=writer_multi,
        args=(
            reader_to_writer_q,
            vis_PORT,
            vis_HOST,
            starttimestamp_q,
            endtimestamp_q,
            delete_flag,
            detect_flag,
            reader_flag,
            activity_flag,
            writer_to_main_q
            ))
    writer_process.daemon = True
    writer_process.start()

    goback_flag = multiprocessing.Event()


    time.sleep(3)

    input("Are you ready? [Press Enter to start this experiment.]")

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    #timer_thread = threading.Thread(
    #    target=timer, args=(
    #        timer_flag, session_length))
    #timer_thread.daemon = True
    #timer_thread.start()

    try:

        while True:

            if word_index == len(transcripts):
                break

            if goback_flag.is_set():
                word_index -= 2
                goback_flag.clear()
                #continue
            transcript = transcripts[word_index]

            #print("now No.",word_index)
            try:
                print(str(word_index),transcript.lower(), pycolor.WHITE + transcripts[word_index+1]+pycolor.END,pycolor.WHITE + transcripts[word_index+2]+pycolor.END,)
            except:
                print(transcript.lower())

            start_flag.wait()
            starttimestamp_q.put(time.time())
            time.sleep(0.2)


            end_flag.wait()
            time.sleep(0.5)
            endtimestamp_q.put(time.time())

            end_flag.clear()
            start_flag.clear()

            smart_palate_frames = writer_to_main_q.get()



            #Check the data
            smart_palate_frames_np = np.array(smart_palate_frames)
            print(smart_palate_frames_np.shape)
            if not smart_palate_frames_np.shape[1] == 124:
                print("data shape error. one more time please.")
                continue

            shimpan = ''
            if shimpan == '':
                #print("saving...")

                data_to_save = {}
                data_to_save["transcript"] = transcript
                data_to_save["smartpalate"] = smart_palate_frames

                with open(dir + str(word_index) + "-"+transcript+".pickle", 'wb') as f:
                    pickle.dump(data_to_save, f)

                #return corresponding_frames
                word_index += 1
                with open(dir + 'word_index.pkl', 'wb') as f:
                    pickle.dump(word_index, f)

    except KeyboardInterrupt:
        print("finishing, thank you!")
        vis_proc.kill()
        exit()
