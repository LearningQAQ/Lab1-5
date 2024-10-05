import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
# Step 1: Generate a synthetic dataset
data = np.random.normal(0, 1, 100)


# Step 2: Define the Gaussian kernel function
def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def epanechnikov_kernel(x):
    return 0.75 * (1 - x ** 2) * (np.abs(x) <= 1)


def uniform_kernel(x):
    return 0.5 * (np.abs(x) <= 1)


def triangular_kernel(x):
    return (1 - np.abs(x)) * (np.abs(x) <= 1)

# Step 3: Define the KDE function
def kde(data, kernel, bandwidth, x):
    n = len(data)
    kde_values = np.zeros_like(x)

    for j in range(len(x)):
        for i in range(n):
            kde_values[j] += kernel((x[j] - data[i]) / bandwidth)
        kde_values[j] /= n * bandwidth
    return kde_values


# Step 4: Plot the estimated density
x = np.linspace(-5, 5, 1000)
bandwidth = 0.3
kde_values = kde(data, gaussian_kernel, bandwidth, x)

plt.figure(figsize=(12, 9))
plt.plot(x, kde_values, label='KDE')
plt.hist(data, bins=10, density=True, label='Histogram of Data')
plt.title('Kernel Density Estimation (KDE)')
plt.legend()
plt.show()

# Part 2
gaussian_part2 = kde(data, gaussian_kernel, bandwidth, x)
epanechnikov_part2 = kde(data, epanechnikov_kernel, bandwidth, x)
uniform_part2 = kde(data, uniform_kernel, bandwidth, x)
triangular_part2 = kde(data, triangular_kernel, bandwidth, x)

# Plot the estimated densities
plt.figure(figsize=(12, 9))
plt.plot(x, gaussian_part2, label='Gaussian KDE', color='red')
plt.plot(x, epanechnikov_part2, label='Epanechnikov KDE', color='green')
plt.plot(x, uniform_part2, label='Uniform KDE', color='orange')
plt.plot(x, triangular_part2, label='Triangular KDE', color='blue')
plt.hist(data, bins=10, density=True, alpha=0.5, label='Histogram of Data')
plt.title('Kernel Density Estimation (KDE) with Different Kernels')
plt.legend()
plt.show()


# Part 3: The Effect of Bandwidth on KDE

# Step 1: Different bandwidth values
bandwidths = [0.1, 0.5, 1, 2]

# Step 2: Plot the resulting KDEs
different_bandwidth = {bw: kde(data, gaussian_kernel, bw, x) for bw in bandwidths}

plt.figure(figsize=(12, 9))
for bw, kde_values in different_bandwidth.items():
    plt.plot(x, kde_values, label=f'Bandwidth = {bw}')
plt.hist(data, bins=10, density=True, label='Histogram of Data')
plt.title('KDE with Different Bandwidths')
plt.legend()
plt.show()

