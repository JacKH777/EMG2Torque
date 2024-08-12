import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, TimeDistributed, Flatten
from save_npy_fun import load_and_print_selected_data, calculate_torque

def build_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=input_shape))
    # 确保 MaxPooling1D 的 pool_size 适应 Conv1D 的输出维度
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    return model
# 数据加载和预处理
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
        x = data_X[i:i+seq_length]
        y = data_y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 25
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
X_test_seq_1, y_test_seq_1 = create_sequences(X_test_scaled_1, y_test_scaled_1, seq_length)

# LSTM 输入需要 3D 的张量 [样本数, 时间步, 特征数]
X_train_seq = np.expand_dims(X_train_seq, axis=-1)
X_test_seq = np.expand_dims(X_test_seq, axis=-1)
X_test_seq_1 = np.expand_dims(X_test_seq_1, axis=-1)

# 确认输入形状
print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"X_test_seq_1 shape: {X_test_seq_1.shape}")

input_shape = (seq_length, 1, 1)  # 时间步长, 特征数

# 构建模型
model = build_cnn_lstm_model(input_shape)
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# 定义 Early Stopping 回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 保存模型
model.save('cnn_lstm_model.h5')

# 加载模型
model = load_model('cnn_lstm_model.h5')

# 进行预测
y_pred_train_seq = model.predict(X_train_seq)
y_pred_test_seq = model.predict(X_test_seq)
y_pred_test_seq_1 = model.predict(X_test_seq_1)

# 反归一化
y_train_rescaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
y_pred_train_rescaled = scaler_y.inverse_transform(y_pred_train_seq)
y_pred_test_rescaled = scaler_y.inverse_transform(y_pred_test_seq)

y_test_rescaled_1 = scaler_y.inverse_transform(y_test_seq_1.reshape(-1, 1))
y_pred_test_rescaled_1 = scaler_y.inverse_transform(y_pred_test_seq_1)

# 平滑预测结果
def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 25
y_pred_train_rescaled_smoothed = smooth(y_pred_train_rescaled.flatten(), window_size)
y_pred_test_rescaled_smoothed = smooth(y_pred_test_rescaled.flatten(), window_size)
y_pred_test_rescaled_smoothed_1 = smooth(y_pred_test_rescaled_1.flatten(), window_size)

# 评估模型
mse_train = mean_squared_error(y_train_rescaled[:len(y_pred_train_rescaled_smoothed)], y_pred_train_rescaled_smoothed)
mse_test = mean_squared_error(y_test_rescaled[:len(y_pred_test_rescaled_smoothed)], y_pred_test_rescaled_smoothed)
mse_test_1 = mean_squared_error(y_test_rescaled_1[:len(y_pred_test_rescaled_smoothed_1)], y_pred_test_rescaled_smoothed_1)

print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print(f'Test MSE 1: {mse_test_1}')

# 绘制训练数据的真实值和预测值
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(y_train_rescaled[:len(y_pred_train_rescaled_smoothed)], label='True Torque (Train)', color='black')
ax1.plot(y_pred_train_rescaled_smoothed, label='Predicted Torque (Train)', color='blue')
ax1.set_title('Torque Prediction (Train)')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Torque')
ax1.legend()

# 绘制测试数据的真实值和预测值
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(y_test_rescaled[:len(y_pred_test_rescaled_smoothed)], label='True Torque (Test)', color='black')
ax2.plot(y_pred_test_rescaled_smoothed, label='Predicted Torque (Test)', color='blue')
ax2.set_title('Torque Prediction (Test)')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Torque')
ax2.legend()

fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(y_test_rescaled_1[:len(y_pred_test_rescaled_smoothed_1)], label='True Torque (Test 1)', color='black')
ax3.plot(y_pred_test_rescaled_smoothed_1, label='Predicted Torque (Test 1)', color='blue')
ax3.set_title('Torque Prediction (Test 1)')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Torque')
ax3.legend()

# 显示图形
plt.show()
