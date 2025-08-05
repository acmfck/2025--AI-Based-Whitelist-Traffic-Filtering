import torch
import torch.nn as nn
import torch.optim as optim
import os
from model.autoencoder import AutoEncoder
from model.lstm_detector import LSTMDetector

from visualization import plot_loss_curve
from data.unsw_nb15_preprocess import load_train_test

train_csv = r"D:\AI\vscode.ai\flow-detector\data\UNSW_NB15_training-set.csv"
test_csv = r"D:\AI\vscode.ai\flow-detector\data\UNSW_NB15_testing-set.csv"
train_loader, test_loader, input_dim, scaler = load_train_test(
    train_csv,
    test_csv,
    batch_size=256,
    drop_service=True  # 如果你想保留 service 列就设为 False
)


print(f"训练集大小: {len(train_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")
use_lstm = True  
# 选择模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = (LSTMDetector(input_dim) if use_lstm else AutoEncoder(input_dim)).to(device)
criterion = nn.MSELoss() if not use_lstm else nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_list = []
test_loss_list = []
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    # 训练步骤
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        if use_lstm:
            if y.dim() > 1:
                y = y.argmax(dim=1)
            y = y.long()
            out = model(x)
            loss = criterion(out, y)
        else:
            out = model(x)
            loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_list.append(total_loss)
    print(f"Loss: {total_loss:.10f}")
    
    
# plot_loss_curve(loss_list)
# 保存模型
torch.save(model.state_dict(), r"D:\AI\vscode.ai\flow-detector\checkpoint3.pt")
print(f"模型已保存到 checkpoint_{use_lstm}.pt")