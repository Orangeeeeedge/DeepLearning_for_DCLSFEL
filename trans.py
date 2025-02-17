import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd

# 加载已保存的归一化
scaler_X = load('scaler_X.pkl')
scaler_Y = load('scaler_y.pkl')

# 加载模型
model = tf.keras.models.load_model('my_model.keras')

# 导入数据
data = pd.read_csv('data_NNGA.csv')
X_in = data.iloc[:, 1:13].values  # 获取特征列
Y_actual = data.iloc[:, -2].values   # 获取标签列

# 步骤1：特征归一化
X_new_scaled = scaler_X.transform(X_in)
# 使用模型进行预测
Y_pred_scaled = model.predict(X_new_scaled)
# 步骤3：反归一化
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# 通过 argsort 排序 Y_actual，并保持对应关系
sorted_indices = np.argsort(Y_actual)  # 获取按 Y_actual 排序的索引

# 使用排序的索引对 X_in 和 Y_pred 进行排序
X_in_sorted = X_in[sorted_indices]
Y_actual_sorted = Y_actual[sorted_indices]
Y_pred_sorted = Y_pred[sorted_indices, -1]  # 获取预测值的最后一列

# 计算排序后的差值
difference_sorted = Y_actual_sorted - Y_pred_sorted

# 可视化排序后的 Y_actual 和 Y_pred
plt.figure(figsize=(10, 6))
plt.plot(Y_pred_sorted, label='Predicted', color='red', linestyle='-', markersize=3)
plt.plot(Y_actual_sorted, label='Actual', color='blue', linestyle='-', markersize=5)
plt.title('Sorted Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)
plt.show()

# 可视化排序后的差值曲线
plt.figure(figsize=(10, 6))
plt.plot(difference_sorted, label='Actual - Predicted Difference', color='green', linestyle='-', markersize=3)
plt.title('Sorted Difference between Actual and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Difference Value')
plt.legend()
plt.grid(True)
plt.show()
