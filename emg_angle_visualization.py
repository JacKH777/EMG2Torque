from eeg_decoder import Decoder, get_latest_date
import numpy as np
import matplotlib.pyplot as plt
import os
from emg_filiter import process_emg_data,downsample_signal,trim_to_match_length
from encoder_function import rescale_array
from save_npy_fun import ensure_directory_exists,save_processed_data

fs = 1000

EXP_DIR   = './exp'
# EXP_DIR   = './EMG_data/yu/other_2kg'
data_date = get_latest_date(EXP_DIR) 
data_date = '2024_08_26_1143'

decoded_file_path = f'{EXP_DIR}/{data_date}/1/1.txt'
decoder = Decoder()
eeg_txt_path = f'{EXP_DIR}/{data_date}/1/EEG.txt'
if (os.path.exists(decoded_file_path)):
    pass
else:
    eeg_data = decoder.decode_to_txt(eeg_txt_path = eeg_txt_path, 
                                        return_data = True,
                                        decode_to_npy = True) 

eeg_raw_path = f'{EXP_DIR}/{data_date}/1/1.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.txt'

angle = np.loadtxt(angle_path)
angle[angle < 20] += 360
angle[angle > 360] = 25
eeg_raw = np.load(eeg_raw_path)
eeg_raw = eeg_raw[5*1000:-1*1000, 2]
# print(len(angle))
# print(len(eeg_raw))
# eeg_raw = eeg_raw[:, 1]

# 添加极端值处理函数
def replace_extreme_values(signal, threshold_factor=3):
    mean_value = np.mean(signal)
    std_value = np.std(signal)
    threshold_high = mean_value + threshold_factor * std_value
    threshold_low = mean_value - threshold_factor * std_value
    
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold_high or signal[i] < threshold_low:
            signal[i] = (signal[i-1] + signal[i+1]) / 2  # 用左右值的平均值替换极端值
    return signal

# 处理EMG数据
eeg_raw = process_emg_data(eeg_raw)
eeg_raw = replace_extreme_values(eeg_raw)  # 替换极端值


# eeg_raw = process_emg_data(eeg_raw)
eeg_raw = downsample_signal(eeg_raw,1000,50)
angle = angle[5*50:-1*50]
# angle = rescale_array(angle,30,90)


eeg_raw,angle = trim_to_match_length(eeg_raw,angle)
t = np.arange(eeg_raw.shape[0]) / 50

# 創建第一個圖形和子圖
fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8))

# 繪製 1000Hz 的信號
axs1[0].plot(t, eeg_raw)
axs1[0].set_title('EMG Data')
axs1[0].set_xlabel('Time [s]')
axs1[0].set_ylabel('Amplitude (V)')
axs1[0].grid(True)

# 繪製 20Hz 的信號
axs1[1].plot(t, angle)
axs1[1].set_title('Angle')
axs1[1].set_xlabel('Time [s]')
axs1[1].set_ylabel('Degree')
axs1[1].grid(True)

# 調整佈局
fig1.tight_layout()

# 顯示所有圖形
# plt.show()

from numpy import diff
from scipy.signal import butter, filtfilt

# 定义移动平均滤波器函数
def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 定义Butterworth低通滤波器函数
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 设定滤波器参数
cutoff_frequency = 2  # 截止频率为24 Hz
sampling_rate = 50  # 采样频率为50 Hz
window_size = 50  # 移动平均窗口大小

# 对角度数据进行移动平均平滑
smoothed_angle = rescale_array(angle,30,90)
smoothed_angle = moving_average_filter(smoothed_angle, window_size)
# smoothed_angle = rescale_array(smoothed_angle,20,90)

# 计算角速度
angular_velocity = np.gradient(smoothed_angle, t)

# 计算角加速度
angular_acceleration = np.gradient(angular_velocity, t)

# 绘制平滑后的角度和角加速度
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 绘制平滑后的角度信号
ax[0].plot(t[150:-150], smoothed_angle[150:-150])
ax[0].set_title('Smoothed Angle')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Angle [degrees]')
ax[0].grid(True)

# 绘制角加速度信号
ax[1].plot(t[150:-150], angular_acceleration[150:-150])
ax[1].set_title('Angular Acceleration')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Acceleration [degree/s^2]')
ax[1].grid(True)

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()