import sys
import os
os.environ["QT_MAC_WANTS_LAYER"] = "1"
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import ss_utils


from socket import socket, AF_INET, SOCK_STREAM
from collections import deque
import time
import threading
import queue


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


args = sys.argv


HOST = 'localhost'
PORT = int(args[1])
MAX_MESSAGE = 2048
NUM_THREAD = 4

CHR_CAN = '\18'
CHR_EOT = '\04'
l_2d_ok = [[0] * 124 for i in range(300)]

q = queue.Queue(maxsize=5000)
d = deque(l_2d_ok, 100)




def receive_from_writer():
    # メッセージ受信ループ
    while True:
        try:
            mess = ss_utils.com_receive(PORT, HOST,socket, AF_INET, SOCK_STREAM)
            # テキスト
            nums_a = [int(str) for str in mess.split()]
            q.put(nums_a)

        except Exception as e:
            print(e)

    print(pycolor.GREEN + 'end of receiver' + pycolor.END)


if __name__ == '__main__':

    # 必ず作らなければいけないオブジェクト
    app = QApplication(sys.argv)

    # Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  # show widget alone in its own window
    win.setWindowTitle('pyqtgraph example: ImageItem')
    view = win.addViewBox()

    # lock the aspect ratio so pixels are always square
    view.setAspectLocked(True)

    # Create image item
    img = pg.ImageItem(border='w')
    view.addItem(img)

    # Set initial view bounds
    view.setRange(QtCore.QRectF(0, 0, 600, 600))

    # Create random image
    data = np.random.normal(
        size=(
            15,
            128,
            100),
        loc=1024,
        scale=128).astype(
            np.uint16)
    i = 0

    img.setImage(data[i])

    updateTime = ptime.time()
    fps = 0

    def imgset(numpied_d):
        global img
        img.setImage(np.array(list(d)))

    def updateData2():
        global img, data, i, updateTime, fps

        if q.full():
            print("blocking")
        # Display the data

        #print("first")

        d.append(q.get(block=True, timeout=None))

        #print("second")

        # print(d)
        imgset_thread = threading.Thread(target=imgset, args=(d, ))

        imgset_thread.start()

        imgset_thread.join(0.3)
        #img.setImage(np.array(list(d)))

        QtCore.QTimer.singleShot(10, updateData2)

        #print("fourth")

    # スレッドの作成とスタート
    receive_thread = threading.Thread(target=receive_from_writer, args=())

    receive_thread.start()
    try:
        updateData2()
        # while True:
        #   updateData3()
        # com_receive()

        # プログラムをクリーンに終了する
    except KeyboardInterrupt:
        print(pycolor.GREEN + "visualizer 異常終了します．" + pycolor.END)

    sys.exit(app.exec_())
