import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 讀取數據和數據預處理
data = pd.read_csv('d:/大三/專題/project/new_spx_40.csv')
features = data[['Open', 'High', 'Low', 'Close']].values

# 設置訓練和測試數據比例
train_size = int(len(features) * 0.8)
train, test = features[:train_size], features[train_size:]

# 標準化數據
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 3])  # 預測 'Close'
    return np.array(X), np.array(Y)

look_back = 5
X_train, y_train = create_dataset(scaled_train, look_back)
X_test, y_test = create_dataset(scaled_test, look_back)

# 轉換為 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最後一個時間步的輸出
        return out

# 初始化模型
input_size = X_train.shape[2]
hidden_size = 70
num_layers = 2
output_size = 1
dropout_rate = 0.5

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)

# 設置優化器與損失函數
learning_rate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

epochs = 100
batch_size = 1

def train_model(model, X_train, y_train, optimizer, criterion, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train):.6f}')

train_model(model, X_train, y_train, optimizer, criterion, epochs, batch_size)

# 預測
def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

train_predict = predict(model, X_train)
test_predict = predict(model, X_test)

# 反標準化預測結果
train_predict = scaler.inverse_transform(
    np.concatenate((train_predict, np.zeros((train_predict.shape[0], features.shape[1] - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(
    np.concatenate((test_predict, np.zeros((test_predict.shape[0], features.shape[1] - 1))), axis=1))[:, 0]

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
