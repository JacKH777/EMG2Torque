import serial
import time, datetime, os, shutil
from datetime import datetime
import keyboard
from decoder_function import decoder

# 初始化串口
ser_1 = serial.Serial("COM4", 460800)
angle = decoder()
angle.get_com_port('COM7')

# 初始化变量
data = ""
total_data = ""
angle_data = ""
angle_total_data = ""
ts = time.time()
data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
fileDir = './exp/{}'.format(data_time)
fileName = 'EEG.txt'
fileName_angle = 'angle.txt'
count = 0

# 创建文件夹
if not os.path.isdir(fileDir):
    os.makedirs(os.path.join(fileDir, '1'))
else:
    shutil.rmtree(fileDir)
    os.makedirs(os.path.join(fileDir, '1'))

try:
    # 打开文件
    with open(os.path.join(fileDir, '1', fileName), 'a') as f_eeg, open(os.path.join(fileDir, '1', fileName_angle), 'a') as f_angle:
        while True:
            try:
                # 读取EEG数据
                data = ser_1.read(32).hex()
                total_data += data

                # 获取角度数据
                # angle_data = str(angle.get_angle()) + " "
                # angle_total_data += angle_data

                count += 1

                if count >= 20:
                    f_eeg.write(total_data)
                    angle_data = str(angle.get_angle()) + " "
                    f_angle.write(angle_data)
                    count = 0
                    total_data = ""
                    angle_total_data = ""

                # 检查是否按下了ESC键
                if keyboard.is_pressed('esc'):
                    f_eeg.write(total_data + '\n')
                    f_angle.write(angle_total_data + '\n')
                    print("ESC pressed, stopping...")
                    break

            except Exception as e:
                print(f"An error occurred while reading or writing data: {e}")
                break

finally:
    # 关闭串口
    ser_1.close()
    # angle.close()
    print("Serial ports closed.")