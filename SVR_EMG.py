import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from save_npy_fun import load_and_print_selected_data, calculate_torque

# 加载和预处理数据
emg_data_train = load_and_print_selected_data("./train_data/jack/", [1, 3, 5], "emg_data.npy")
angle_data_train = load_and_print_selected_data("./train_data/jack/", [1, 3, 5], "angle_data.npy")
weight = 1  # 例如，重物的重量为1千克
torque_train = np.array([calculate_torque(weight, angle) for angle in angle_data_train])

emg_data_test = np.load('./train_data/jack/2024_07_30_1334/emg_data.npy')
angle_data_test = np.load('./train_data/jack/2024_07_30_1334/angle_data.npy')
torque_test = np.array([calculate_torque(weight, angle) for angle in angle_data_test])

emg_data_test_1 = np.load('./train_data/jack/2024_08_02_0952/emg_data.npy')
angle_data_test_1 = np.load('./train_data/jack/2024_08_02_0952/angle_data.npy')
torque_test_1 = np.array([calculate_torque(weight, angle) for angle in angle_data_test_1])

# 数据归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = emg_data_train
y_train = torque_train
X_test = emg_data_test
y_test = torque_test
X_test_1 = emg_data_test_1
y_test_1 = torque_test_1

X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(-1)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(-1)
X_test_scaled_1 = scaler_X.transform(X_test_1.reshape(-1, 1)).reshape(-1)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
y_test_scaled_1 = scaler_y.transform(y_test_1.reshape(-1, 1)).reshape(-1)

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
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
X_test_seq_1, y_test_seq_1 = create_sequences(X_test_scaled_1, y_test_scaled_1, seq_length)

# 构建 SVR 模型
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 训练模型
svr_model.fit(X_train_seq, y_train_seq)

# 进行预测
y_pred_train_seq = svr_model.predict(X_train_seq)
y_pred_test_seq = svr_model.predict(X_test_seq)
y_pred_test_seq_1 = svr_model.predict(X_test_seq_1)

# 反归一化
y_train_rescaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
y_pred_train_rescaled = scaler_y.inverse_transform(y_pred_train_seq.reshape(-1, 1))
y_pred_test_rescaled = scaler_y.inverse_transform(y_pred_test_seq.reshape(-1, 1))

y_test_rescaled_1 = scaler_y.inverse_transform(y_test_seq_1.reshape(-1, 1))
y_pred_test_rescaled_1 = scaler_y.inverse_transform(y_pred_test_seq_1.reshape(-1, 1))

# 平滑预测结果
def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 10
y_pred_train_rescaled_smoothed = smooth(y_pred_train_rescaled.flatten(), window_size)
y_pred_test_rescaled_smoothed = smooth(y_pred_test_rescaled.flatten(), window_size)
y_pred_test_rescaled_smoothed_1 = smooth(y_pred_test_rescaled_1.flatten(), window_size)

# 评估模型
mse_train = mean_squared_error(y_train_rescaled, y_pred_train_rescaled)
mse_test = mean_squared_error(y_test_rescaled, y_pred_test_rescaled)
mse_test_1 = mean_squared_error(y_test_rescaled_1, y_pred_test_rescaled_1)

print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print(f'Test MSE 1: {mse_test_1}')

# 绘制训练数据的真实值和预测值
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(y_train_rescaled, label='True Torque (Train)', color='black')
ax1.plot(y_pred_train_rescaled, label='Predicted Torque (Train)', color='blue')
ax1.set_title('Torque Prediction (Train)')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Torque')
ax1.legend()

# 绘制测试数据的真实值和预测值
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(y_test_rescaled, label='True Torque (Test)', color='black')
ax2.plot(y_pred_test_rescaled, label='Predicted Torque (Test)', color='blue')
ax2.set_title('Torque Prediction (Test)')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Torque')
ax2.legend()

# 绘制第二个测试数据的真实值和预测值
fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(y_test_rescaled_1, label='True Torque (Test 1)', color='black')
ax3.plot(y_pred_test_rescaled_1, label='Predicted Torque (Test 1)', color='blue')
ax3.set_title('Torque Prediction (Test 1)')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Torque')
ax3.legend()

# 显示图形
plt.show()
