import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, hilbert,decimate,medfilt

def apply_bandpass_filter(data, fs, lowcut=10.0, highcut=100, order=6):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_hilbert_transform(data):
    analytic_signal = hilbert(data)
    return np.abs(analytic_signal)

def apply_lowpass_filter(data, fs, cutoff=0.3, order=3):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def apply_moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def apply_median_filter(data, kernel_size):
    return medfilt(data, kernel_size)

def  process_emg_data(raw_data, fs=1000.0):
    if raw_data.shape[0] < 10:
        raise ValueError("The length of the input vector must be greater than 9.")
    
    
    # 应用带通滤波器
    filtered_data = apply_bandpass_filter(raw_data, fs)

    # 应用全波整流
    rectified_data = apply_hilbert_transform(filtered_data)
    # rectified_data = apply_full_wave_rectification(filtered_data)

    # 应用低通滤波器取包络线
    envelope_data = apply_lowpass_filter(rectified_data, fs)
    smoothed_data = apply_median_filter(envelope_data, 501)

    return envelope_data

def downsample_signal(signal, original_rate, target_rate):
    """
    将一个信号从原始采样率降采样到目标采样率。

    参数:
    signal (np.ndarray): 原始信号
    original_rate (int): 原始采样率
    target_rate (int): 目标采样率

    返回:
    np.ndarray: 降采样后的信号
    """
    if target_rate >= original_rate:
        raise ValueError("目标采样率必须小于原始采样率")
    
    # 计算降采样因子
    decimation_factor = original_rate // target_rate
    
    # 使用 SciPy 的 decimate 函数进行降采样
    downsampled_signal = decimate(signal, decimation_factor)
    
    return downsampled_signal

def trim_to_match_length(array1, array2):
    """
    将两个数组裁剪到相同长度（取较小的长度），从较大的数组末尾去除多余的元素。

    参数:
    array1 (np.ndarray): 第一个数组
    array2 (np.ndarray): 第二个数组

    返回:
    np.ndarray, np.ndarray: 裁剪后的两个数组
    """
    # 找到较小的长度
    min_length = min(len(array1), len(array2))
    
    # 裁剪数组
    trimmed_array1 = array1[:min_length]
    trimmed_array2 = array2[:min_length]
    
    return trimmed_array1, trimmed_array2