import serial
import time, datetime, os, shutil
from datetime import datetime
import threading
import keyboard
from encoder_function import encoder
from eeg_decoder import Decoder, Filter
import numpy as np

# 初始化串口
ser_1 = serial.Serial("COM4", 460800)
angle = encoder()
angle.get_com_port('COM7')

decoder = Decoder()
filter = Filter()

# 全局变量和锁
buffer_size = 2000 * 64   # 缓冲区大小，保持一定数量的数据（乘以2是因为使用字符串保存十六进制，1字节 -> 2字符）
buffer = ""  # 使用字符串作为环形缓冲区
data_to_process = 0  # 记录新接收到的数据量

buffer_lock = threading.Lock()  # 用于同步访问共享缓冲区的锁
event = threading.Event()

# 文件相关设置
ts = time.time()
data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
fileDir = './exp/{}'.format(data_time)
fileName = 'EEG.txt'
fileName_angle = 'angle.txt'
fileName_filtered = 'filtered_EEG.txt'  # 保存滤波后的数据

# 创建文件夹
if not os.path.isdir(fileDir):
    os.makedirs(os.path.join(fileDir, '1'))
else:
    shutil.rmtree(fileDir)
    os.makedirs(os.path.join(fileDir, '1'))

# 数据接收线程
def receive_data():
    global buffer
    total_data = ""
    count = 0
    try:
        with open(os.path.join(fileDir, '1', fileName), 'a') as f_eeg, open(os.path.join(fileDir, '1', fileName_angle), 'a') as f_angle:
            while True:
                try:
                    # 读取32字节的EEG数据，并转换为十六进制字符串
                    data = ser_1.read(32).hex()

                    if data:
                        total_data += data
                        # 使用锁确保线程安全修改 buffer
                        with buffer_lock:
                            buffer += data  # 将数据追加到缓冲区
                            if len(buffer) > buffer_size:
                                # 如果缓冲区超过最大大小，丢弃最旧的数据
                                buffer = buffer[64:]


                        count += 1

                        # 保存数据文件，假设每20次保存一次
                        if count % 20 == 0:
                            f_eeg.write(total_data)
                            angle_data = str(angle.get_angle()) + " "
                            f_angle.write(angle_data + '\n')
                            total_data = ""
                        
                        if count > 500:
                            event.set()
                            count = 0

                    # 检查是否按下ESC键
                    if keyboard.is_pressed('esc'):
                        f_eeg.write(data + '\n')
                        f_angle.write(angle_data + '\n')
                        print("ESC pressed, stopping...")
                        break

                except Exception as e:
                    print(f"An error occurred while receiving data: {e}")
                    break
    finally:
        ser_1.close()
        print("Serial ports closed.")

# 滤波线程
def process_data():
    global buffer
    with open(os.path.join(fileDir, '1', fileName_filtered), 'a') as f_filtered:
        while True:
            event.wait() 
            local_buffer = ""
            
            # 使用锁从全局字符串缓冲区复制数据到本地变量
            with buffer_lock:
                local_buffer = buffer

            if len(local_buffer) >= buffer_size:
                # 解码数据
                decoded_data = decoder.decode(local_buffer, show_progress=False)
                filtered_data = decoded_data[1000:,1]

                # # 取最后一个滤波结果，并写入文件
                # filtered_value = filtered_data[-1, 1]  # 假设我们需要第二列数据
                np.savetxt(f_filtered, filtered_data, delimiter=',')
                # f_filtered.write(f"{filtered_value:.6f}\n")
                # f_filtered.flush()
            event.clear()


# 启动接收数据的线程
receive_thread = threading.Thread(target=receive_data)
receive_thread.daemon = True
receive_thread.start()

# 启动数据处理（滤波）线程
process_thread = threading.Thread(target=process_data)
process_thread.daemon = True
process_thread.start()

# 等待接收数据线程结束
receive_thread.join()