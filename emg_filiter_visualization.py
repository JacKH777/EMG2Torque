import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt, hilbert, lfilter

class EEGProcessor:
    def __init__(self, fs):
        self.fs = fs

    def butter_bandpass(self, lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandstop(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='stop')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=6):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

def apply_lowpass_filter(data, fs, cutoff=3.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

# 設置取樣率
fs = 1000  

# 讀取數據
eeg_raw_path = "C:/Users/Jack/Documents/ncu/code/EMG/exp/2024_07_11_2221/1/1.npy"
eeg_raw = np.load(eeg_raw_path)

# 截取中間部分數據並選擇第3個通道 (從0開始計數，所以下標為2)
eeg_raw = eeg_raw[1000:-1000, 2]

# 生成時間向量
t = np.arange(eeg_raw.shape[0]) / fs

# 創建EEGProcessor對象
processor = EEGProcessor(fs)

# 濾波處理
eeg_filtered = processor.butter_bandstop_filter(eeg_raw, 55, 65, fs)
eeg_filtered = processor.butter_bandpass_filter(eeg_filtered, 5, 100, fs)

# 畫原始數據圖
plt.figure(1,figsize=(12, 6))
plt.plot(t, eeg_raw, label='Raw EMG')
plt.title('Raw EMG')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (V)')

# 畫濾波後數據圖
plt.figure(2,figsize=(12, 6))
plt.plot(t[650:], eeg_filtered[650:])
# plt.plot(t, eeg_filtered)
plt.title('55Hz~65Hz Bandstop, 5Hz~100Hz Bandpass EMG')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (V)')


hilbert_emg = hilbert(eeg_filtered[650:])
# hilbert_emg = np.real(hilbert_emg) + np.imag(hilbert_emg)
plt.figure(3,figsize=(12, 6))
plt.plot(t[650:], np.abs(hilbert_emg))
plt.title('Absolute Hilbert Transform EMG')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (V)')


# abs_emg = np.abs(hilbert_emg)
# plt.figure(4,figsize=(12, 6))
# plt.plot(t[650:], abs_emg)
# plt.title('Absolute EMG')
# plt.xlabel('Time (sec)')
# plt.ylabel('Amplitude (V)')

emg_3hz = apply_lowpass_filter(np.abs(hilbert_emg), fs, cutoff=1.0, order=3)
plt.figure(5,figsize=(12, 6))
plt.plot(t[650:], emg_3hz)
plt.title('1Hz Lowpass EMG')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.show()