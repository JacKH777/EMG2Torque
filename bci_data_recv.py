# 2021/02/04 EEG_button
#  https://stackoverflow.com/a/6981055/6622587
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
# from Ui_NewGUI import Ui_MainWindow
# from trash import Ui_MainWindow
from bci_gui import Ui_MainWindow
import shutil
import sys
import multiprocessing
import serial
import time
import numpy as np
from datetime import datetime
import os

from eeg_decoder import Decoder, Filter
from multiprocessing import Queue



# https://www.pythonguis.com/tutorials/plotting-matplotlib/
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# from EEGModels import EEGNet
from scipy import signal

import serial.tools.list_ports

# from emg_filiter import process_emg_data




def styled_text(text=None, color="#999999"):
    if text is None:
        text = datetime.now().strftime("%H:%M:%S")                
    return f"<span style=\" font-size:8pt; color:{color};\" >" + text + "</span>"    


class MyMainWindow(QtWidgets. QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Brain GUI')
        
        # 按鍵功能
        self.btnCon.clicked.connect(self.StartConnection)  # 連線
        self.btnDisCon.clicked.connect(self.Disconnection)  # 斷線
        self.btnSave.clicked.connect(self.Savedata)  # 存檔


        # 多線程
        self.queue_data_save_flag = Queue()
        self.queue_plot_data = Queue()
        self.queue_model_data = Queue()
        self.queue_comport = Queue()
        self.queue_gui_message = Queue()

        # 建立資料接收class
        self.dt = DataReceiveThreads()  

        # 多線程 : 開始接收封包
        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, 
                                                      args=(self.queue_comport, self.queue_plot_data, 
                                                            self.queue_data_save_flag, self.queue_gui_message)) 


        self.decoder = Decoder()
        self.raw_total = ""            


        # ------------------------------------ #
        # Show all COM Port in combobox 
        # ------------------------------------ #
        default_idx = -1
        ports = serial.tools.list_ports.comports()
        for i, port in enumerate(ports):
            # port.device = 'COMX'
            if "USB 序列裝置" in port.description:
                default_idx = i
                self.queue_comport.put(port.device)
                print(f"Selected default COM : {port.description}")

                self.message.append(styled_text())                
                self.message.append(f'>> Default COM : {port.device}')

            self.comboBox.addItem(port.device + ' - ' + port.description)
        
        self.comboBox.setCurrentIndex(default_idx)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)

        self.timer_activate = False


    def update_plot(self):
        raw = self.queue_plot_data.get()
        # clear the queue
        while not self.queue_plot_data.empty():
            temp = self.queue_plot_data.get() 
            del temp
             

        # # 64 * 3000 = 192000
        if len(self.raw_total) >= 384000: 
            self.raw_total = self.raw_total[-64000:]
        else:
            self.raw_total = self.raw_total + raw
        
        # shape = (n, 10)
        eeg_raw = np.array(self.decoder.decode(self.raw_total))         
        ydata = eeg_raw[1000:-1].T       
        xdata = np.arange(ydata.shape[1])
        # if len(ydata[1]) > 9:
        #     ydata[1] = process_emg_data(ydata[0])
        for i in range(8):
            if i == 0:
                self.canvas.lines[0].set_data(xdata, ydata[1])
            else:
                break
            # if i == 1:
            #     ydata[i] = process_emg_data(ydata[i])
            # self.canvas.lines[i].set_data(xdata, ydata[i])
        # for i in range(2):
        #     self.canvas.lines[i].set_data(xdata, ydata[i])
        
        self.canvas.draw()



    def update_time(self):
        elapsed_time = time.time() - self.start_time
        time_text = f"{(elapsed_time//60):>02.0f} : {(elapsed_time%60):>04.1f} ({elapsed_time:>04.1f})"
 
        # showing it to the label
        self.label_time.setText(time_text)



    def on_combobox_changed(self, index):
        if index < 0:
            return
        # 取得選擇的 COM Port
        COM_PORT = self.comboBox.itemText(index).split(' ')[0]
        print(f'Selected Port: {COM_PORT}')
        self.queue_comport.put(COM_PORT)


        self.message.append(styled_text()) 
        self.message.append(f'>> Selected Port: {COM_PORT}')
        

    def StartConnection(self):  
        # 連線        
        self.multipDataRecv.start()
        self.queue_data_save_flag.put(False)


        while True:
            if not self.queue_gui_message.empty():
                # Get last selected COM port name from queue
                message = self.queue_gui_message.get()

                self.message.append(styled_text()) 
                self.message.append(f'>> {message}')                
                break
        
        
    def Disconnection(self):
        if not self.multipDataRecv.is_alive():            
            print ("Process has not started")
        else:
            self.message.append(styled_text()) 
            self.message.append(f'>> All Processes had been\nterminated. You can close\nthis window.')


            print("All processes had been killed\nYou can close this window")
            self.multipDataRecv.terminate()
            self.queue_data_save_flag.put(False)

        if self.timer_activate:
            self.timer.stop()
            self.timer_2.stop()


    def Savedata(self):
        self.message.append(styled_text())      
        self.message.append(f'>> Start data streaming')
        self.queue_data_save_flag.put(True)
        
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)   
        self.timer_activate = True           


        self.start_time = time.time()
        self.timer_2 = QtCore.QTimer()
        self.timer_2.timeout.connect(self.update_time)
        self.timer_2.start(10)   
      

class DataReceiveThreads(Ui_MainWindow):
    def __init__(self):
        self.if_save = "0"
        self.data = ""
        self.count = 0
        self.total_data = ""

        self.count_model = 0
        self.total_data_model = ""
        self.small_data = ""

        # 創立當前時間的txt檔案
        ts = time.time()
        data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
        self.fileDir = './exp/{}'.format(data_time)

        if not os.path.isdir(self.fileDir):
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        else:
            shutil.rmtree(self.fileDir)
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        
        self.fileName = 'EEG.txt'


    def data_recv(self, queue_comport, queue_plot_data, queue_data_save_flag, queue_gui_message):
        while True:            
            if not queue_comport.empty():
                # Get last selected COM port name from queue
                COM_PORT = queue_comport.get()
                break
        

        print(f"Open {COM_PORT}...")
        ser = serial.Serial(COM_PORT, 460800)
        print(f"Successfull Receive")
        queue_gui_message.put("Successfull Receive!\nReady to data streaming")


        while True:
            ser.reset_output_buffer() 
            ser.reset_input_buffer()            
            if not queue_data_save_flag.empty():
                self.save_flag = queue_data_save_flag.get()
            if self.save_flag:
                break
    
        f = open(f"{self.fileDir}/1/{self.fileName}", "a")

        while True:     
            if not queue_data_save_flag.empty():
                self.save_flag = queue_data_save_flag.get()
            if not self.save_flag:
                # 結束後寫入最後收到的資料到EEG.txt
                with open(f"{self.fileDir}/1/{self.fileName}", "a") as f:
                    f.write(self.total_data)
                    f.close()
                
            # 每次讀取 32 bytes(一組EEG data的大小)並轉成16進位。收一次等於 1ms 的時間
            self.data = ser.read(32).hex() 
            self.total_data = self.total_data + self.data
            self.count = self.count + 1
            
            # -------------------------------------------------------- #
            # 送去畫圖的資料 (每 100 ms 寫入資料到txt的最尾端)
            # -------------------------------------------------------- #                                    
            if self.count >= 100:
                queue_plot_data.put(self.total_data)

                f.write(self.total_data)
                self.count = 0
                self.total_data = ""                




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())    