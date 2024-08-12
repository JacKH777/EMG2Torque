import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from eeg_decoder import Decoder, get_latest_date
from emg_filiter import process_emg_data, downsample_signal, trim_to_match_length
from decoder_function import rescale_array
from save_npy_fun import ensure_directory_exists, save_processed_data


fs = 1000

# EXP_DIR = './exp'
EXP_DIR = './EMG_data/jack/other_2kg'
# data_date = get_latest_date(EXP_DIR)
data_date = '2024_08_08_1954'

decoded_file_path = f'{EXP_DIR}/{data_date}/1/1.txt'
decoder = Decoder()
eeg_txt_path = f'{EXP_DIR}/{data_date}/1/EEG.txt'
if os.path.exists(decoded_file_path):
    pass
else:
    eeg_data = decoder.decode_to_txt(eeg_txt_path=eeg_txt_path,
                                     return_data=True,
                                     decode_to_npy=True)

eeg_raw_path = f'{EXP_DIR}/{data_date}/1/1.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.txt'
angle = np.loadtxt(angle_path)
angle[angle < 20] += 360
eeg_raw = np.load(eeg_raw_path)
eeg_raw = eeg_raw[:, 2]

# 数据预处理
eeg_raw = process_emg_data(eeg_raw)
eeg_raw = downsample_signal(eeg_raw, 1000, 50)

eeg_raw, angle = trim_to_match_length(eeg_raw, angle)
t = np.arange(eeg_raw.shape[0]) / 50

# 调整角度数据到0-360度范围内
angle[angle < 20] += 360

# 找到所有接近30度的点
indices_30 = np.where((angle >= 20) & (angle <= 30))[0]

# 计算每个点的斜率
slopes = np.gradient(angle)

# 筛选出接近30度且在200点范围内具有最大斜率的点
selected_points = []
for idx in indices_30:
    start_idx = max(idx - 100, 0)
    end_idx = min(idx + 100, len(slopes))
    if end_idx - start_idx > 0:
        max_slope_idx = start_idx + np.argmax(np.abs(slopes[start_idx:end_idx]))
        if 20 <= angle[max_slope_idx] <= 30:
            if not selected_points or max_slope_idx - selected_points[-1] >= 200:
                selected_points.append(max_slope_idx)
            else:
                if np.abs(slopes[max_slope_idx]) > np.abs(slopes[selected_points[-1]]):
                    selected_points[-1] = max_slope_idx

# 扩展标记点之间的范围
all_marked_points = set(selected_points)
for i in range(1, len(selected_points)):
    start = selected_points[i - 1]
    end = selected_points[i]
    all_marked_points.update(np.where((angle[start:end] >= 20) & (angle[start:end] <= 30))[0] + start)

# 排序所有标记点
all_marked_points = sorted(all_marked_points)

# 修改segments[0]让它从起点且平滑的接到segments[0]的末端
segments = []
current_segment = [0]
for i in range(len(all_marked_points)):
    current_segment.append(all_marked_points[i])
    if i < len(all_marked_points) - 1 and all_marked_points[i + 1] - all_marked_points[i] > 1:
        segments.append(current_segment)
        current_segment = [all_marked_points[i + 1]]
segments.append(current_segment)

# 打印段数
print(f"Number of segments: {len(segments)}")

# 只保留第一个 segment 的后 150 个点
if len(segments[0]) > 150:
    first_segment_start = segments[0][-151]  # 包含索引在内，共150个点的起始位置
else:
    first_segment_start = segments[0][0]  # 如果segment小于150，则从segment的起点开始

# 从第一个 segment 的后 150 个点开始裁剪数据
eeg_raw = eeg_raw[first_segment_start:]
angle = angle[first_segment_start:]
t = t[first_segment_start:]

# 更新 selected_points 和 segments 的索引
selected_points = [p - first_segment_start for p in selected_points if p >= first_segment_start]
segments = [[p - first_segment_start for p in segment if p >= first_segment_start] for segment in segments]

# 只保留最后一个 segment 的前 150 个点
if len(segments[-1]) > 150:
    segments[-1] = segments[-1][:150]
else:
    segments[-1] = segments[-1]  # 如果segment小于150，则保留整个segment

# 更新数据范围
last_segment_end = segments[-1][-1]

eeg_raw = eeg_raw[:last_segment_end + 1]
angle = angle[:last_segment_end + 1]
t = t[:last_segment_end + 1]

# 更新 selected_points 的范围，防止超出索引
selected_points = [p for p in selected_points if p <= last_segment_end]

# 使用非线性平滑方法（指数衰减）连接每个段
extended_angles = angle.copy()

def smooth_transition_with_slope(start_idx, end_idx, start_val, end_val, start_slope, end_slope, state):
    length = end_idx - start_idx + 1
    if state == "up":
        x = np.linspace(1, 0, length)
        transition = (start_val - end_val) * (1 - np.exp(-5 * x)) + end_val
        transition_slope_adjusted = transition + (start_slope * (1 - x) + end_slope * x) * (1 - np.exp(-7 * x))
    else:
        x = np.linspace(0, 1, length)
        transition = (start_val - end_val) * np.exp(-5 * x) + end_val
        transition_slope_adjusted = transition + (start_slope * x + end_slope * (1 - x)) * np.exp(-7 * x)
    return transition_slope_adjusted

segment_0_start = 0
segment_0_end = segments[0][-1]
start_slope_0 = slopes[segment_0_start] if segment_0_start < len(slopes) else 0
end_slope_0 = slopes[segment_0_end] if segment_0_end < len(slopes) else 0
extended_angles[segment_0_start:segment_0_end + 1] = smooth_transition_with_slope(segment_0_start, segment_0_end, 0, angle[segment_0_end], start_slope_0, end_slope_0, "up")

for segment in segments[1:-1]:
    segment_start = segment[0]
    segment_end = segment[-1]
    segment_length = segment_end - segment_start + 1
    mid_idx = segment_start + segment_length // 2

    start_slope = slopes[segment_start] if segment_start < len(slopes) else 0
    end_slope = slopes[segment_end] if segment_end < len(slopes) else 0

    smooth_transition_start = smooth_transition_with_slope(segment_start, mid_idx, angle[segment_start], 0, start_slope, 0, "down")
    smooth_transition_end = smooth_transition_with_slope(mid_idx, segment_end, 0, angle[segment_end], 0, end_slope, "up")

    extended_angles[segment_start:mid_idx + 1] = smooth_transition_start
    extended_angles[mid_idx:segment_end + 1] = smooth_transition_end

segment_last_start = segments[-1][0]
segment_last_end = segments[-1][-1]
start_slope_last = slopes[segment_last_start] if segment_last_start < len(slopes) else 0
end_slope_last = slopes[segment_last_end] if segment_last_end < len(slopes) else 0
extended_angles[segment_last_start:segment_last_end + 1] = smooth_transition_with_slope(segment_last_start, segment_last_end, angle[segment_last_start], 0, start_slope_last, end_slope_last, "down")

fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8))

axs1[0].plot(t, eeg_raw)
axs1[0].set_title('EMG Data')
axs1[0].set_xlabel('Time [s]')
axs1[0].set_ylabel('Amplitude (V)')
axs1[0].grid(True)

axs1[1].plot(t, angle, label='Original Angle')
axs1[1].plot(t, extended_angles, label='Extended Angle')
if selected_points:
    axs1[1].plot(t[selected_points], angle[selected_points], 'ro', label='Significant Points', markersize=3)
for segment in segments:
    if segment:
        axs1[1].plot(t[segment[0]:segment[-1] + 1], angle[segment[0]:segment[-1] + 1], 'ro', markersize=3)
axs1[1].set_title('Angle')
axs1[1].set_xlabel('Time [s]')
axs1[1].set_ylabel('Degree')
axs1[1].grid(True)
axs1[1].legend()

fig1.tight_layout()

fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8))

m = 2
g = 9.81
d_load = 0.14
torque = g * d_load * np.sin(extended_angles / 180 * np.pi)  * m

axs2[0].plot(t, eeg_raw)
axs2[0].set_title('EMG Data')
axs2[0].set_xlabel('Time [s]')
axs2[0].set_ylabel('Amplitude (V)')
axs2[0].grid(True)

axs2[1].plot(t, torque)
axs2[1].set_title('Torque')
axs2[1].set_xlabel('Time [s]')
axs2[1].set_ylabel('N*m')
axs2[1].grid(True)

fig2.tight_layout()

# plt.show()

# 使用中点切割数据
mid_indices = [segment[0] + len(segment) // 2 for segment in segments if len(segment) > 1]

# 根据中点切割数据
torque_subarrays = []
emg_subarrays = []
start_idx = 0

for mid_idx in mid_indices:
    if start_idx < mid_idx:  # 检查起始索引是否小于中点索引
        torque_subarrays.append(torque[start_idx:mid_idx])
        emg_subarrays.append(eeg_raw[start_idx:mid_idx])
    start_idx = mid_idx

# 选取前 10 段数据并将它们连接在一起
num_segments_to_plot = 11
selected_emg_data = []
selected_torque_data = []
selected_time_data = []

for i in range(min(num_segments_to_plot, len(torque_subarrays))):
    torque_subarray = torque_subarrays[i]
    emg_subarray = emg_subarrays[i]
    segment_time = np.arange(len(torque_subarray)) / 50  # 假设每个样本点间隔时间为 1/50 秒
    
    selected_emg_data.extend(emg_subarray)
    selected_torque_data.extend(torque_subarray)
    if selected_time_data:
        selected_time_data.extend(segment_time + selected_time_data[-1] + (1/50))  # 保持时间连续
    else:
        selected_time_data.extend(segment_time)

# 将数据连接后绘制成一个图
fig4, axs4 = plt.subplots(2, 1, figsize=(12, 8))

axs4[0].plot(selected_time_data, selected_emg_data, label='EMG Data')
axs4[0].set_title('EMG Data')
axs4[0].set_xlabel('Time [s]')
axs4[0].set_ylabel('Amplitude (V)')
axs4[0].grid(True)

axs4[1].plot(selected_time_data, selected_torque_data, label='Torque')
axs4[1].set_title('Torque')
axs4[1].set_xlabel('Time [s]')
axs4[1].set_ylabel('N*m')
axs4[1].grid(True)

fig4.tight_layout()
plt.show()

# 保存数据函数
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_processed_data(data, filename, path):
    full_path = os.path.join(path, filename)
    np.save(full_path, data)
    print(f"数据已保存至: {full_path}")

# 询问用户是否要保存数据
save_data = input("是否要保存数据? (y/n): ")

if save_data.lower() == 'y':
    subject_name = input("受试者名称: ")
    subject_test = input("资料: ")
    train_data_path = f"./train_data/{subject_name}/{subject_test}/{data_date}"
    ensure_directory_exists(train_data_path)
    
    for i, (torque_subarray, emg_subarray) in enumerate(zip(torque_subarrays, emg_subarrays)):
        if i == 13:
            break
        if len(torque_subarray) > 0 and len(emg_subarray) > 0:  # 确保数组非空
            save_processed_data(torque_subarray, f'torque_{i+1}.npy', train_data_path)
            save_processed_data(emg_subarray, f'emg_{i+1}.npy', train_data_path)
        else:
            print(f"Skipped empty array at index {i}")

else:
    print("数据未保存")
