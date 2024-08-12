import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from eeg_decoder import get_latest_date

# 假设你的数据已经加载到变量 emg 和 torque 中
EXP_DIR = './exp'
# data_date = get_latest_date(EXP_DIR)
data_date = '2024_07_19_0406'
emg = np.load(f'{EXP_DIR}/{data_date}/1/emg.npy')
torque = np.load(f'{EXP_DIR}/{data_date}/1/torque.npy')
downsample_factor = 1000 // 20

# 直接取每隔downsample_factor个样本点
emg_data_downsampled = emg[::downsample_factor]

# 准备数据，合并 emg 和 torque 为一个 DataFrame
data = pd.DataFrame({'emg': emg_data_downsampled, 'torque': torque})

# 归一化数据
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 拆分特征和标签
X = data[['emg']]
y = data[['torque']]

# 分别进行归一化
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练和测试数据
train_size_4 = int(len(X_scaled) * 0.4)
train_size_5 = int(len(X_scaled) * 0.5)
X_train = np.concatenate((X_scaled[:train_size_4], X_scaled[train_size_5:train_size_5 + train_size_4]), axis=0)
X_test = np.concatenate((X_scaled[train_size_4:train_size_5], X_scaled[train_size_5 + train_size_4:]), axis=0)
y_train = np.concatenate((y_scaled[:train_size_4], y_scaled[train_size_5:train_size_5 + train_size_4]), axis=0)
y_test = np.concatenate((y_scaled[train_size_4:train_size_5], y_scaled[train_size_5 + train_size_4:]), axis=0)

# 切割数据函数
def create_sequences(data_X, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_X) - seq_length):
        x = data_X[i:i+seq_length].flatten()  # 将数据展平
        y = data_y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# 构建 AdaBoost 模型
adaboost_model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=3), 
    n_estimators=100,  # 增加基学习器的数量
    learning_rate=0.05  # 减小学习率
)

# 训练模型
adaboost_model.fit(X_train_seq, y_train_seq)

# 进行预测
y_pred_train_seq = adaboost_model.predict(X_train_seq)
y_pred_test_seq = adaboost_model.predict(X_test_seq)

# 反归一化
y_train_rescaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
y_pred_train_rescaled = scaler_y.inverse_transform(y_pred_train_seq.reshape(-1, 1))
y_pred_test_rescaled = scaler_y.inverse_transform(y_pred_test_seq.reshape(-1, 1))

# 平滑预测结果函数
def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 是否应用平滑处理
apply_smoothing = True
window_size = 5

if apply_smoothing:
    y_pred_train_rescaled_smoothed = smooth(y_pred_train_rescaled.flatten(), window_size)
    y_pred_test_rescaled_smoothed = smooth(y_pred_test_rescaled.flatten(), window_size)
    y_train_rescaled = y_train_rescaled[window_size-1:]
    y_test_rescaled = y_test_rescaled[window_size-1:]
else:
    y_pred_train_rescaled_smoothed = y_pred_train_rescaled.flatten()
    y_pred_test_rescaled_smoothed = y_pred_test_rescaled.flatten()

# 评估模型
mse_train = mean_squared_error(y_train_rescaled, y_pred_train_rescaled_smoothed)
mse_test = mean_squared_error(y_test_rescaled, y_pred_test_rescaled_smoothed)
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')

# 绘制结果
plt.figure(figsize=(14, 5))

# 绘制训练数据的真实值和预测值
plt.plot(y_train_rescaled, label='True Torque', color='black')
plt.plot(np.arange(len(y_train_rescaled)), y_pred_train_rescaled_smoothed, label='Predicted Torque', color='blue')

# 绘制测试数据的真实值和预测值
plt.plot(np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), y_test_rescaled, color='black')
plt.plot(np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_pred_test_rescaled_smoothed)), y_pred_test_rescaled_smoothed, color='blue')

# 添加虚线分隔
plt.axvline(x=len(y_train_rescaled), color='gray', linestyle='--')
plt.text(len(y_train_rescaled) / 2, plt.ylim()[1], 'Train', horizontalalignment='center', fontsize=12, color='red')
plt.text(len(y_train_rescaled) + len(y_test_rescaled) / 2, plt.ylim()[1], 'Test', horizontalalignment='center', fontsize=12, color='red')
plt.legend()

# 动态设置标题
title_suffix = 'with Moving Average Smoothing' if apply_smoothing else 'without Smoothing'
plt.title(f'Torque Prediction (AdaBoost {title_suffix})')
plt.xlabel('Time Step')
plt.ylabel('Torque')
plt.show()
