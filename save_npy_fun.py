import os
import numpy as np

def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建它。

    参数:
    directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_processed_data(data, filename, directory):
    """
    保存处理好的数据到指定目录中的文件。

    参数:
    data (np.ndarray): 要保存的数据
    filename (str): 文件名
    directory (str): 目录路径
    """
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, filename)
    np.save(file_path, data)
    print(f"Data saved to {file_path}")


def load_and_print_selected_data(file_directory, selected_indices,npy_name):
    """
    从指定目录中加载指定索引的numpy文件数据并打印。

    参数:
    file_directory (str): 存放文件的目录路径
    selected_indices (list): 需要加载的文件索引（从1开始）

    返回:
    list: 加载的数据列表
    """
    # 获取目录中所有文件名，并按名称排序（假设文件名为日期格式）
    file_list = sorted([f for f in os.listdir(file_directory)])
    selected_data = np.array([])

    for index in selected_indices:
        # 文件名索引从1开始，但列表索引从0开始，因此减1
        dir_name = file_list[index - 1]
        file_path = os.path.join(file_directory, dir_name)
        npy_path = os.path.join(file_path, npy_name)
        
        if not os.path.exists(npy_path):
            print(f"文件 {npy_path} 不存在。")
            continue
        
        data = np.load(npy_path)
        
        if selected_data.size == 0:
            selected_data = data
        else:
            selected_data = np.concatenate((selected_data, data), axis=0)
        print(f"Data from file {npy_path}")
    
    
    return selected_data


def calculate_torque(weight, angle_degrees):
    """
    计算力矩。

    参数:
    force (float): 施加的力（单位：牛顿）
    distance (float): 力作用点到旋转轴的距离（单位：米）
    angle_degrees (float): 力作用方向与旋转轴之间的角度（单位：度）

    返回:
    float: 力矩（单位：牛米）
    """
    distance = 0.14

    force = weight * 9.81

    # 将角度从度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 计算力矩
    torque = force * distance * np.sin(angle_radians)
    
    return torque

def find_local_extrema(arr):
    local_max = (np.diff(np.sign(np.diff(arr))) < 0).nonzero()[0] + 1
    local_min = (np.diff(np.sign(np.diff(arr))) > 0).nonzero()[0] + 1
    return local_max, local_min

