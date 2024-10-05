import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Step 1:加载 Iris 数据集
iris = load_iris()
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_features.to_csv('iris.csv', index=False)


# Step 2：
# Define the Gaussian kernel
def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def epanechnikov_kernel(x):
    return 0.75 * (1 - x ** 2) * (np.abs(x) <= 1)


def uniform_kernel(x):
    return 0.5 * (np.abs(x) <= 1)


def triangular_kernel(x):
    return (1 - np.abs(x)) * (np.abs(x) <= 1)


# Define the KDE function
def kde(data, kernel, bandwidth, x):
    n = len(data)
    kde_values = np.zeros_like(x)

    for j in range(len(x)):
        for i in range(n):
            kde_values[j] += kernel((x[j] - data[i]) / bandwidth)
        kde_values[j] /= n * bandwidth
    return kde_values


df = pd.read_csv('iris.csv')
sepal_width = df['sepal width (cm)']

x = np.linspace(sepal_width.min() - 1, sepal_width.max() + 1, 1000)
bandwidth = 0.2

# Step 2: 计算不同核的 KDE
gaussian = kde(sepal_width, gaussian_kernel, bandwidth, x)
epanechnikov = kde(sepal_width, epanechnikov_kernel, bandwidth, x)
uniform = kde(sepal_width, uniform_kernel, bandwidth, x)
triangular = kde(sepal_width, triangular_kernel, bandwidth, x)

# Step 3: 可视化不同核的 KDE
plt.figure(figsize=(12, 6))
plt.plot(x, gaussian, label='Gaussian KDE')
plt.plot(x, epanechnikov, label='Epanechnikov KDE')
plt.plot(x, uniform, label='Uniform KDE')
plt.plot(x, triangular, label='Triangular KDE')
plt.hist(sepal_width, bins=10, density=True,label='Histogram of Sepal Width')
plt.title('KDE of Sepal Width with Different Kernels')
plt.legend()
plt.show()

# 对于Iris数据集中的“Sepal Width”这种经典的生物测量数据，Gaussian 核可能是最合适的选择。
# 它能够生成平滑的曲线，适合处理像Iris数据集这样有自然波动的数据，尤其是分布呈现较规则的钟形时。
# Gaussian核通常在各种情况下表现较稳健，因此是一个可靠的选择。


# gaussian kde
plt.figure(figsize=(12, 6))
plt.plot(x, gaussian, label='Gaussian KDE')
plt.hist(sepal_width, bins=10, density=True,label='Histogram of Sepal Width')
plt.title('KDE of Sepal Width with Different Kernels')
plt.legend()
plt.show()