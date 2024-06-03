import os
from keras.layers import Dropout, Dense, LSTM, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
data = pd.read_csv("SH600519.csv")

# 使用多个特征
features = ['open', 'close', 'high', 'low', 'volume']
training_set = data.iloc[0:2126, 2:6].values
test_set = data.iloc[2126:, 2:6].values

# 数据归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

# 创建数据集
def create_dataset(dataset, time_step=60):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, :])
        y.append(dataset[i, 1])  
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(training_set)
x_test, y_test = create_dataset(test_set)

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 调整输入的形状为 [样本数, 时间步长, 特征数]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# 构建LSTM模型
model = tf.keras.Sequential([
    SimpleRNN(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.1),
    SimpleRNN(100),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mean_squared_error')

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=100, 
                    validation_data=(x_test, y_test), 
                    callbacks=[early_stopping]
                    )

# 绘制训练和验证损失
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 预测
predicted_stock_price = model.predict(x_test)

# 反归一化预测值，只针对 'close' 列
sc_close = MinMaxScaler(feature_range=(0, 1))
sc_close.min_, sc_close.scale_ = sc.min_[1], sc.scale_[1]
predicted_stock_price = sc_close.inverse_transform(predicted_stock_price)
real_stock_price = sc_close.inverse_transform(y_test.reshape(-1, 1))

# 绘制真实值和预测值
plt.plot(real_stock_price, color='red', label='Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
