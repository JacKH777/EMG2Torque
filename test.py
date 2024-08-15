import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
import time, datetime, os, shutil
from datetime import datetime
from eeg_decoder import Decoder
from encoder_function import decoder
import threading
import queue
from collections import deque

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # 创建主窗口的小部件
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)

        # 创建垂直布局
        self.layout = QtWidgets.QVBoxLayout(self.centralWidget)

        # 创建两个绘图部件
        self.graphWidget_emg = pg.PlotWidget()
        self.graphWidget_angle = pg.PlotWidget()

        # 添加绘图部件到布局
        self.layout.addWidget(self.graphWidget_emg)
        self.layout.addWidget(self.graphWidget_angle)

        # 设置EMG图表
        self.graphWidget_emg.setBackground('w')
        self.graphWidget_emg.setTitle("Real-time EMG Data", color="b", size="20pt")
        styles = {"color": "#f00", "font-size": "15px"}
        self.graphWidget_emg.setLabel("left", "Amplitude", **styles)
        self.graphWidget_emg.setLabel("bottom", "Time", **styles)
        self.graphWidget_emg.showGrid(x=True, y=True)

        # 设置角度图表
        self.graphWidget_angle.setBackground('w')
        self.graphWidget_angle.setTitle("Real-time Angle Data", color="b", size="20pt")
        self.graphWidget_angle.setLabel("left", "Angle", **styles)
        self.graphWidget_angle.setLabel("bottom", "Time", **styles)
        self.graphWidget_angle.showGrid(x=True, y=True)

        # 初始化数据
        self.x = list(range(250))  # 初始化X轴数据
        self.y_emg = [0] * 250  # 初始化Y轴数据（EMG）
        self.y_angle = [0] * 250  # 初始化Y轴数据（角度）

        # 绘制初始数据
        self.data_line_emg = self.graphWidget_emg.plot(self.x, self.y_emg, pen=pg.mkPen(color=(255, 0, 0), width=2), name="EMG")
        self.data_line_angle = self.graphWidget_angle.plot(self.x, self.y_angle, pen=pg.mkPen(color=(0, 0, 255), width=2), name="Angle")

        # 初始化串口
        try:
            self.ser_1 = serial.Serial("COM4", 460800)
        except serial.SerialException as e:
            print(f"An error occurred while opening the serial port: {e}")
            sys.exit(1)

        self.decoder = Decoder()
        self.angle = decoder()
        self.angle.get_com_port('COM7')

        # 初始化文件保存
        ts = time.time()
        data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
        self.fileDir = './exp/{}'.format(data_time)
        if not os.path.isdir(self.fileDir):
            os.makedirs(os.path.join(self.fileDir, '1'))
        else:
            shutil.rmtree(self.fileDir)
            os.makedirs(os.path.join(self.fileDir, '1'))

        self.fileName_raw = os.path.join(self.fileDir, '1', 'EEG.txt')
        self.fileName_angle = os.path.join(self.fileDir, '1', 'angle.txt')
        self.raw_data_buffer_save = ""
        self.angle_total_data_save = ""

        # 初始化队列用于线程间通信
        self.angle_queue = queue.Queue()
        # self.filtered_emg_queue = queue.Queue()  # 用于存储滤波后的EMG数据

        # 初始化环形缓冲区用于存储最近6秒的EMG数据（假设采样率为1000Hz）
        self.emg_circular_buffer = deque(maxlen=6001*64)  # 6秒 * 1000Hz * 32字节

        # 初始化线程锁
        self.lock = threading.Lock()

        # 启动线程读取EMG和角度数据
        self.emg_thread = threading.Thread(target=self.read_emg_data)
        self.angle_thread = threading.Thread(target=self.read_angle_data)
        # self.filter_thread = threading.Thread(target=self.filter_emg_data)
        self.file_write_thread = threading.Thread(target=self.write_to_file)
        self.emg_thread.daemon = True
        self.angle_thread.daemon = True
        # self.filter_thread.daemon = True
        self.file_write_thread.daemon = True
        self.emg_thread.start()
        self.angle_thread.start()
        # self.filter_thread.start()
        self.file_write_thread.start()

        # 设置定时器来更新图表数据
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # 每50毫秒更新一次图表
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def read_emg_data(self):
        try:
            emg_count = 0
            while True:
                try:
                    raw_data = self.ser_1.read(32).hex()  # 读取32字节EMG数据
                    self.raw_data_buffer_save += raw_data
                    # with self.lock:
                    #     self.emg_circular_buffer.append(raw_data)
                    # emg_count += 1
                    # if emg_count > 6000:  # 6秒的数据对应的字节数
                    #     with self.lock:
                    #         self.emg_circular_buffer = deque(list(self.emg_circular_buffer)[-6000:], maxlen=6001*64)  # 保留最後的6000個元素
                    #     emg_count = 0
                except serial.SerialException as e:
                    print(f"An error occurred while reading EMG data: {e}")
                    break
        except Exception as e:
            print(f"An error occurred in the read_emg_data thread: {e}")

    def read_angle_data(self):
        try:
            while True:
                try:
                    angle_value = self.angle.get_angle()
                    self.angle_total_data_save += str(angle_value) + " "
                    self.angle_queue.put(angle_value)
                    # time.sleep(0.02)  # 每20毫秒读取一次角度数据
                except Exception as e:
                    print(f"An error occurred while reading angle data: {e}")
                    break
        except Exception as e:
            print(f"An error occurred in the read_angle_data thread: {e}")

    # def filter_emg_data(self):
    #     try:
    #         while True:
    #             raw_data_combined = ""
    #             with self.lock:
    #                 if len(self.emg_circular_buffer) > 39:
    #                     raw_data_list = list(self.emg_circular_buffer)  # 将缓冲区转换为列表
    #                     raw_data_combined = ''.join(raw_data_list)  # 合并缓冲区中的数据
    #             if raw_data_combined:  # 确保 raw_data_combined 非空
    #                 decoded_data = self.decoder.decode(raw_data_combined)  # 对缓冲区进行解码
    #                 filtered_data = decoded_data[:, 1]  # 对解码后的数据进行滤波处理
    #                 self.filtered_emg_queue.put(filtered_data[-1])  # 将滤波后的最后一个值放入队列
    #     except Exception as e:
    #         print(f"An error occurred in the filter_emg_data thread: {e}")

    def write_to_file(self):
        try:
            while True:
                if self.raw_data_buffer_save:
                    with open(self.fileName_raw, 'a') as f:
                        f.write(self.raw_data_buffer_save)
                    self.raw_data_buffer_save = ""

                if self.angle_total_data_save:
                    with open(self.fileName_angle, 'a') as f:
                        f.write(self.angle_total_data_save)
                    self.angle_total_data_save = ""

                time.sleep(1)  # 每秒写入一次
        except Exception as e:
            print(f"An error occurred in the write_to_file thread: {e}")

    def update_plot_data(self):
        try:
            # # 更新EMG数据
            # if not self.filtered_emg_queue.empty():
            #     emg_value = self.filtered_emg_queue.get()
            #     while not self.filtered_emg_queue.empty():
            #         self.filtered_emg_queue.get()

                # # 更新数据
                # self.x = self.x[1:]  # 移除第一个元素
                # self.x.append(self.x[-1] + 1)  # 增加一个新元素

                # self.y_emg = self.y_emg[1:]  # 移除第一个元素
                # self.y_emg.append(emg_value)  # 增加一个新元素

            # 更新角度数据
            if not self.angle_queue.empty():
                angle_value = self.angle_queue.get()
                while not self.angle_queue.empty():
                    self.angle_queue.get()

                self.y_angle = self.y_angle[1:]  # 移除第一个元素
                self.y_angle.append(angle_value)  # 增加一个新元素

            # # 更新图表
            # self.data_line_emg.setData(self.x, self.y_emg)
            self.data_line_angle.setData(self.x, self.y_angle)

        except Exception as e:
            print(f"An error occurred while updating plot data: {e}")

    def closeEvent(self, event):
        try:
            self.ser_1.close()
        except Exception as e:
            print(f"An error occurred while closing the serial port: {e}")

        try:
            self.f_raw.write(self.raw_data_buffer_save)
            self.f_angle.write(self.angle_total_data_save)
            self.f_raw.close()
            self.f_angle.close()
        except Exception as e:
            print(f"An error occurred while closing files: {e}")

        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
