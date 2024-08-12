import serial
from scipy.signal import butter, filtfilt

class decoder():
    def __init__(self):
        self.ser = None
        self.first_count = 0
        self.reset_count = 0

    def get_com_port(self,ser_com):
        self.ser = serial.Serial(ser_com, 115200)

    def get_angle(self):
        self.ser.write(b'\x54')
        read_data = self.ser.read(2)
        received_val = int.from_bytes(read_data, byteorder='little')

        # 将整数转换成二进制，并移除最高两位
        binary_val = bin(received_val)[2:].zfill(16)  # 将整数转换为16位的二进制字符串
        truncated_binary = binary_val[2:]  # 移除最高两位
        actual_angle = int(truncated_binary, 2)

        # 校正
        if self.first_count==0:
            self.reset_count = actual_angle
            self.first_count = 1
        if actual_angle < 8192 and self.reset_count > 8192:
            actual_angle = ((actual_angle-self.reset_count+16383)/16383*360)+25
        elif actual_angle > 8192 and self.reset_count > 8192:
            actual_angle = ((actual_angle-self.reset_count)/16383*360)+25
        elif actual_angle < 8192 and self.reset_count < 8192:
            actual_angle = ((actual_angle-self.reset_count)/16383*360)+25
        else:
            actual_angle = ((actual_angle-self.reset_count-16383)/16383*360)+25
        # if actual_angle > 350:
        #     actual_angle = 28
        return actual_angle

def rescale_array(array, new_min, new_max):
    """
    将一个数组的值线性转换到指定的新范围内。

    参数:
    array (np.ndarray): 原始数组
    new_min (float): 新范围的最小值
    new_max (float): 新范围的最大值

    返回:
    np.ndarray: 转换后的新数组
    """
    # 计算原始数组的最小值和最大值
    min_original = array.min()
    max_original = array.max()

    # 线性转换公式
    new_array = ((array - min_original) / (max_original - min_original)) * (new_max - new_min) + new_min

    return new_array

# 設計Butterworth濾波器
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 前向-後向濾波函數
def forward_backward_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)  # 使用filtfilt函數進行前向-後向濾波
    return y

