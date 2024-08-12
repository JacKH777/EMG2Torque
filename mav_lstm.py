import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from keras.regularizers import l2

# 定义一个函数来递归遍历目录并加载数据，并按比例分割数据
def load_data_from_directory(directory, exclude_dirs=[], train_ratio=0.8):
    train_emg_data_list = []
    train_torque_data_list = []
    test_emg_data_list = []
    test_torque_data_list = []

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        emg_files = [file for file in files if 'emg' in file and file.endswith('.npy')]
        torque_files = [file for file in files if 'torque' in file and file.endswith('.npy')]
        
        # 将数据文件配对
        paired_files = list(zip(emg_files, torque_files))
        
        # 按照比例分割为训练集和测试集
        train_size = int(len(paired_files) * train_ratio)
        train_files = paired_files[:train_size]
        test_files = paired_files[train_size:]
        
        # 加载训练集数据
        for emg_file, torque_file in train_files:
            emg_data_path = os.path.join(root, emg_file)
            torque_data_path = os.path.join(root, torque_file)
            if os.path.exists(emg_data_path) and os.path.exists(torque_data_path):
                emg_data = np.load(emg_data_path)
                torque_data = np.load(torque_data_path)

                if emg_data.size > 0 and torque_data.size > 0:
                    train_emg_data_list.append(emg_data)
                    train_torque_data_list.append(torque_data)

        # 加载测试集数据
        for emg_file, torque_file in test_files:
            emg_data_path = os.path.join(root, emg_file)
            torque_data_path = os.path.join(root, torque_file)
            if os.path.exists(emg_data_path) and os.path.exists(torque_data_path):
                emg_data = np.load(emg_data_path)
                torque_data = np.load(torque_data_path)

                if emg_data.size > 0 and torque_data.size > 0:
                    test_emg_data_list.append(emg_data)
                    test_torque_data_list.append(torque_data)

    return train_emg_data_list, train_torque_data_list, test_emg_data_list, test_torque_data_list

# 创建时间序列并计算平均值
def create_sequences(data_X, data_y, seq_length):
    xs, ys = [], []
    for i in range(len(data_X) - seq_length):
        x = data_X[i:i+seq_length]
        y = data_y[i+seq_length]
        x_avg = np.mean(x)  # 求时间窗口内的平均值
        xs.append(x_avg)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 预处理数据
def preprocess_data(emg_data_list, torque_data_list, scaler_X, scaler_y, seq_length):
    X_seq_list = []
    y_seq_list = []

    for emg_data, torque_data in zip(emg_data_list, torque_data_list):
        if emg_data.size == 0 or torque_data.size == 0:
            continue
        
        # 分别进行归一化
        X_scaled = scaler_X.transform(emg_data.reshape(-1, 1)).reshape(-1)
        y_scaled = scaler_y.transform(torque_data.reshape(-1, 1)).reshape(-1)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
        X_seq_list.append(X_seq)
        y_seq_list.append(y_seq)

    if len(X_seq_list) == 0 or len(y_seq_list) == 0:
        return np.array([]), np.array([])

    return np.concatenate(X_seq_list, axis=0), np.concatenate(y_seq_list, axis=0)

# 构建双向LSTM模型
def build_bidirectional_lstm_model(input_shape, l2_lambda=0.01):
    model = Sequential()
    
    # 添加L2正则化和批量归一化到LSTM层
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu', 
                                 kernel_regularizer=l2(l2_lambda)), 
                            input_shape=input_shape))
    model.add(BatchNormalization())  # 添加批量归一化
    model.add(Dropout(rate=0.3))
    
    model.add(Bidirectional(LSTM(32, activation='relu', 
                                 kernel_regularizer=l2(l2_lambda))))
    model.add(BatchNormalization())  # 添加批量归一化
    model.add(Dropout(rate=0.3))
    
    # 添加L2正则化到Dense层
    model.add(Dense(1, kernel_regularizer=l2(l2_lambda)))
    return model

# 加载数据
data_directory = "./train_data/jack/"
seq_length = 20

# 先将数据按比例切分
train_ratio = 0.8
train_emg_data_list, train_torque_data_list, test_emg_data_list, test_torque_data_list = load_data_from_directory(data_directory, ["2kg", "other_2kg"], train_ratio)

# 打乱训练集和测试集的数据顺序
train_indices = np.arange(len(train_emg_data_list))
np.random.shuffle(train_indices)
train_emg_data_list = [train_emg_data_list[i] for i in train_indices]
train_torque_data_list = [train_torque_data_list[i] for i in train_indices]

test_indices = np.arange(len(test_emg_data_list))
np.random.shuffle(test_indices)
test_emg_data_list = [test_emg_data_list[i] for i in test_indices]
test_torque_data_list = [test_torque_data_list[i] for i in test_indices]

# 将所有数据合并用于计算全局的最小值和最大值
all_emg_data = np.concatenate(train_emg_data_list + test_emg_data_list)
all_torque_data = np.concatenate(train_torque_data_list + test_torque_data_list)

# 初始化Scaler并计算全局的最小值和最大值
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(all_emg_data.reshape(-1, 1))
scaler_y.fit(all_torque_data.reshape(-1, 1))

# 对训练和测试数据进行预处理
X_train_seq, y_train_seq = preprocess_data(train_emg_data_list, train_torque_data_list, scaler_X, scaler_y, seq_length)
X_test_seq, y_test_seq = preprocess_data(test_emg_data_list, test_torque_data_list, scaler_X, scaler_y, seq_length)

# 重新调整数据形状以适应LSTM的输入格式
X_train_seq = X_train_seq.reshape(-1, seq_length, 1)
X_test_seq = X_test_seq.reshape(-1, seq_length, 1)

# 确认输入形状
print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")

input_shape = (seq_length, 1)  # 时间步长和特征数

# 选择模型
model = build_bidirectional_lstm_model(input_shape)

# 编译并训练模型
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mae')

# 定义 Early Stopping 回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 进行预测
y_pred_train_seq = model.predict(X_train_seq)
y_pred_test_seq = model.predict(X_test_seq)

# 反归一化
def inverse_transform(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

y_train_rescaled = inverse_transform(y_train_seq, scaler_y)
y_test_rescaled = inverse_transform(y_test_seq, scaler_y)
y_pred_train_rescaled = inverse_transform(y_pred_train_seq, scaler_y)
y_pred_test_rescaled = inverse_transform(y_pred_test_seq, scaler_y)

# 平滑预测结果
def smooth(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 25
y_pred_train_rescaled_smoothed = smooth(y_pred_train_rescaled, window_size)
y_pred_test_rescaled_smoothed = smooth(y_pred_test_rescaled, window_size)

# 评估模型
mse_train = mean_squared_error(y_train_rescaled[:len(y_pred_train_rescaled_smoothed)], y_pred_train_rescaled_smoothed)
mse_test = mean_squared_error(y_test_rescaled[:len(y_pred_test_rescaled_smoothed)], y_pred_test_rescaled_smoothed)
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')

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

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

# 绘制损失变化图
plot_loss(history)
# 显示图形
plt.show()
