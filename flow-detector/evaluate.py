import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score

from model.autoencoder import AutoEncoder
from model.lstm_detector import LSTMDetector
from data.unsw_nb15_preprocess import load_train_test

# ===== 配置区域 =====
use_lstm = True  # 与训练保持一致
train_csv = r"D:\AI\vscode.ai\flow-detector\data\UNSW_NB15_training-set.csv"
test_csv = r"D:\AI\vscode.ai\flow-detector\data\UNSW_NB15_testing-set.csv"
train_loader, test_loader, input_dim, scaler = load_train_test(
    train_csv,
    test_csv,
    batch_size=256,
    drop_service=True  # 如果你想保留 service 列就设为 False
)
model_path = r"D:\AI\vscode.ai\flow-detector\checkpoint3.pt"  # 训练保存的模型路径
threshold = 0.01  # AE 阈值判断用
# ===================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = (LSTMDetector(input_dim) if use_lstm else AutoEncoder(input_dim)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 收集测试结果
y_true, y_pred, y_score = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if use_lstm:
            if y.dim() > 1:
                y = y.argmax(dim=1)
            out = model(x)
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)[:, 1]  # 预测为1的概率
        else:
            recon = model(x)
            loss = ((x - recon) ** 2).mean(dim=1)
            pred = (loss > threshold).int()
            prob = loss  # reconstruction loss 可用于排序

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_score.extend(prob.cpu().numpy())

# 输出评估指标
print("=== 评估结果 ===")
print(classification_report(y_true, y_pred, digits=4))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_true, y_score):.4f}")
# save_path = os.path.join("D:/AI/vscode.ai/flow-detector", "est_checkpoint1.pt")
# torch.save(model.state_dict(), save_path)