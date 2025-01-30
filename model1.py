import pandas as pd #848 3179
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# 讀取數據和數據預處理
data = pd.read_csv('d:/大三/專題/project/new_spx_40.csv')
# 假設數據中有 'Open', 'High', 'Low', 'Close' 列
features = data[['Open', 'High', 'Low', 'Close']].values
# 設置訓練和測試數據比例
train_size = int(len(features) * 0.8)
train, test = features[:train_size], features[train_size:]
# 標準化數據
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)  # 僅用訓練數據擬合
scaled_test = scaler.transform(test)  # 使用訓練數據的參數轉換測試數據
# 創建數據集函數
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 3])  # 使用 'Close' 作為預測目標
    return np.array(X), np.array(Y)
look_back = 5
X_train, y_train = create_dataset(scaled_train, look_back)
X_test, y_test = create_dataset(scaled_test, look_back)
# 調整數據形狀以適應 LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
# 構建 LSTM 模型
model = Sequential()
model.add(LSTM(50,activation='relu', input_shape=(look_back, X_train.shape[2],), kernel_regularizer=l2(0.1),return_sequences=True))
model.add(Dropout(0.75))
model.add(LSTM(50, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(Dropout(0.5))
model.add(Dense(1))
# 設置固定的學習率
learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)
# 編譯模型時使用指定的優化器
model.compile(optimizer=optimizer, loss='mean_squared_error')
# 訓練模型並保存歷史記錄
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, validation_split=0.2, callbacks=[early_stopping])
# 繪製學習曲線
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
# 預測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# 反標準化預測結果
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], features.shape[1] - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], features.shape[1] - 1))), axis=1))[:, 0]
# 計算 RMSE
train_score = np.sqrt(mean_squared_error(y_train, train_predict))
test_score = np.sqrt(mean_squared_error(y_test, test_predict))
print(f'Train RMSE: {train_score:.2f}')
print(f'Test RMSE: {test_score:.2f}')

# 繪製結果
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Close Price', color='blue')
plt.plot(range(look_back + len(X_train), look_back + len(X_train) + len(test_predict)), test_predict, label='Test Predict', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
