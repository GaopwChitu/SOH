# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% 读取预测数据文件
# 读取predistion.csv，假设文件包含三列：cycle（循环次数）、real_SoH（真实SoH值）、pred_SoH（预测SoH值）
# 若你的列名不同，请自行修改
df = pd.read_csv('predictions.csv')

# 提取关键数据列
cycle = df['cycle'].values  # 循环次数
real_SoH = df['true'].values  # 真实SoH值
pred_SoH = df['pred'].values  # 预测SoH值

# %% 划分前50%为已用真实数据，后50%为测试对比数据
total_size = len(df)
train_size = int(total_size * 0.5)  # 前50%作为used real data

# 划分数据
# 已用真实数据（前50%）
cycle1 = cycle[:train_size]
used_real_SoH = real_SoH[:train_size]

# 测试数据（后50%：真实值+预测值）
cycle2 = cycle[train_size:]
test_real_SoH = real_SoH[train_size:]
test_pred_SoH = pred_SoH[train_size:]

# %% 打印数据形状信息
print("Shape of total cycle :", cycle.shape)
print("Shape of total real SoH :", real_SoH.shape)
print("Shape of total prediction SoH :", pred_SoH.shape)
print("Shape of used real data (cycle1) :", cycle1.shape)
print("Shape of test real data :", test_real_SoH.shape)
print("Shape of test prediction data :", test_pred_SoH.shape)

# %% 计算测试集的RMSE和MAE
rmse = math.sqrt(mean_squared_error(test_real_SoH, test_pred_SoH))
mae = mean_absolute_error(test_real_SoH, test_pred_SoH)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)

# %% md
# ## 4. Visualization
# %% 绘制SoH预测可视化图
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

# 绘制已用真实数据（前50%）
plt.plot(cycle1, used_real_SoH, label='Used real data', linewidth=3, color='r')
# 绘制测试集真实数据
plt.plot(cycle2, test_real_SoH, label='Real data', linewidth=3, color='b')
# 绘制测试集预测数据
plt.plot(cycle2, test_pred_SoH, label='TCN-LSTM Prediction', linewidth=3, color='g')

# 设置图例、坐标轴标签和标题
plt.legend(prop={'size': 16})
plt.ylabel('SoH', fontsize=15)
plt.xlabel('Discharge cycle', fontsize=15)
# 此处num[0]可根据你的实际需求修改，若没有num变量可直接写固定名称
plt.title("B05 SOH Prediction", fontsize=15)

# 保存图片（保持原有保存路径格式）
plt.savefig('TCN_LSTM_Prediction.jpg')
plt.show()