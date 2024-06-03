import os,math
from keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data = pd.read_csv("SH600519.csv")  # 读取股票文件
training_set = data.iloc[0:2426 - 300, 2:3].values  


test_set = data.iloc[2426 - 300:, 2:3].values  
sc           = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(training_set)
test_set     = sc.transform(test_set) 
x_train = []
y_train = []

x_test = []
y_test = []
for i in range(1, len(training_set)):
    x_train.append(training_set[i - 1:i, 0])
    y_train.append(training_set[i, 0])
    
for i in range(1, len(test_set)):
    x_test.append(test_set[i - 1:i, 0])
    y_test.append(test_set[i, 0])
    
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
x_train, y_train = np.array(x_train), np.array(y_train) 
x_test,  y_test  = np.array(x_test),  np.array(y_test)

"""
输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
"""
x_train = np.reshape(x_train, (x_train.shape[0], 1, 1))
x_test  = np.reshape(x_test,  (x_test.shape[0], 1, 1))

model = tf.keras.Sequential([
    SimpleRNN(100, return_sequences=True), #布尔值。是返回输出序列中的最后一个输出，还是全部序列。
    Dropout(0.1),                         #防止过拟合
    SimpleRNN(100),
    Dropout(0.1),
    Dense(1)
])


model2 = tf.keras.Sequential([
    LSTM(100, return_sequences=True),
    Dropout(0.1),
    LSTM(100),
    Dropout(0.1),
    Dense(1)
])

model3 = tf.keras.Sequential([
    GRU(100, return_sequences=True),
    Dropout(0.1),
    GRU(100),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=20, 
                    validation_data=(x_test, y_test), 
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()

plt.plot(history.history['loss']    , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
predicted_stock_price = model.predict(x_test)
#predicted_stock_price = predicted_stock_price.reshape(predicted_stock_price.shape[0], -1) # Reshape to 2D
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(test_set[1:])              # 对真实数据还原---从（0，1）反归一化到原始范围

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()