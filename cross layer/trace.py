import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d


mu, sigma = 5, 4  # 均值和标准差
samples = np.random.normal(mu, sigma, 2000)

# 插值以增加样本数（oversampling rate为3）
x_original = np.arange(2000)
x_interpolated = np.linspace(0, 1999, 2000 * 20)  # 3倍oversampling
interpolated_samples = interp1d(x_original, samples, kind='linear')(x_interpolated)+ np.random.normal(0,0.6,len(x_interpolated))

# 将小于0的值设为0.1，大于8的值设为7.9
interpolated_samples[interpolated_samples < 5] = 5 + np.random.normal(0,0.5)
interpolated_samples[interpolated_samples > 9] = 9 + np.random.normal(0,0.6)

positive_samples = interpolated_samples
# a = np.load('pose.npy')
# positive_samples = a[20:]
print(positive_samples[0:40])

# # 处理插值后的高斯序列使得所有值都大于零
# positive_samples = interpolated_samples - np.min(interpolated_samples) + 0.01


# 第一种预测方法：前五个样本的调和平均作为预测值
predicted_samples = []
for i in range(4, len(positive_samples)):
    harmonic_mean = hmean(positive_samples[i-4:i+1])
    predicted_samples.append(harmonic_mean)

# 第二种预测方法：每4个样本平均，然后用5个历史平均样本的调和平均预测
averaged_samples = [np.mean(positive_samples[i:i+4]) for i in range(0, len(positive_samples), 4)]
predicted_averaged_samples = []
for i in range(4, len(averaged_samples)):
    harmonic_mean = hmean(averaged_samples[i-4:i+1])
    predicted_averaged_samples.append(harmonic_mean)

# 扩展第二种方法的预测值以计算MSE
expanded_predicted_averaged_samples = []
for val in predicted_averaged_samples:
    expanded_predicted_averaged_samples.extend([val] * 4)

# 计算MSE
mse_method1 = mean_squared_error(positive_samples[4:100], predicted_samples[:96])
mse_method2 = mean_squared_error(positive_samples[16:100], expanded_predicted_averaged_samples[:84])
print(mse_method1, mse_method2)

# 画图
plt.figure(figsize=(15, 6))
plt.plot(positive_samples[:500], label='Processed Real Sequence', alpha=0.7)
plt.plot(range(4, 500), predicted_samples[:496], label='Method 1 Predicted Sequence', alpha=0.7)
plt.plot(range(16, 500), expanded_predicted_averaged_samples[:484], label='Method 2 Predicted Sequence', alpha=0.7)
plt.title('Comparison of Two Prediction Methods with Real Sequence (First 500 Samples)')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.legend()
plt.show()



