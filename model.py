from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten,Bidirectional,Dropout,SimpleRNN,BatchNormalization # type: ignore
from keras.regularizers import l2

def build_cnn_lstm_model(input_shape):
    model = Sequential()
    # Conv1D expects input_shape to be 3D: (timesteps, features, 1)
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    return model

def build_bidirectional_lstm_model(input_shape, l2_lambda=0.01):
    model = Sequential()
    
    # 添加L2正则化和批量归一化到LSTM层
    model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())  # 添加批量归一化
    model.add(Dropout(rate=0.3))
    
    model.add(Bidirectional(LSTM(8)))
    model.add(BatchNormalization())  # 添加批量归一化
    model.add(Dropout(rate=0.3))

    # 添加L2正则化到Dense层
    model.add(Dense(1))
    return model

def build_simple_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(16, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(rate=0.3))
    model.add(SimpleRNN(8))
    model.add(Dense(1))
    return model

def build_lstm_model(input_shape, l2_lambda=0.01):
    model = Sequential()
    
    # 第一层 LSTM
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_lambda), input_shape=input_shape))
    model.add(BatchNormalization())  # 批量归一化
    model.add(Dropout(rate=0.3))     # Dropout
    
    # 第二层 LSTM
    model.add(LSTM(32, kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())  # 批量归一化
    model.add(Dropout(rate=0.3))     # Dropout
    
    # 全连接层
    model.add(Dense(1))  # 输出层
    
    return model