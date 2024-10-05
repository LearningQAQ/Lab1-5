import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

df = pd.read_excel('decathlete2008_dataLabo4.xlsx', sheet_name='Points')
# print(df.head())
y = (df['Overall'] - df['Overall'].mean()) / df['Overall'].std()  #bia

events_columns = ['Run100_pts', 'LJ_pts', 'SP_pts', 'HJ_pts', 'Run400_pts', 'H_pts', 'DT_pts', 'PV_pts', 'JT_pts', 'Run1500_pts']
df = df[events_columns]
print(df.to_string())

# correlation heatmap
sns.heatmap(df.corr(), annot=True)

vif = pd.Series([variance_inflation_factor(df.values, df.columns.get_loc(i)) for i in df.columns], index=df.columns)
print(vif)


# 2. 标准化数据
df_scaled = (df - df.mean()) / df.std()

# 3. 计算协方差矩阵的特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(df_scaled.cov())


# 4. 对特征值和特征向量按特征值降序排序
sorted_idx = np.argsort(eig_vals)[::-1]  # 降序排序索引
eig_vals = eig_vals[sorted_idx]  # 按照降序重新排序特征值
eig_vecs = eig_vecs[:, sorted_idx]  # 对应重新排序特征向量


# 4. 计算总方差
total_var = df_scaled.var().sum()
print(f"Total variance: {total_var:.2f}")

# 5. 计算特征值的总和
eig_vals_sum = eig_vals.sum()
print(f"Sum of eigenvalues: {eig_vals_sum:.2f}")

# 验证特征值的总和与总方差相等
assert eig_vals_sum.round(2) == total_var.round(2)

# 6. 计算每个主成分解释的方差比例
var_exp = eig_vals / total_var
print(f"Proportion of variance explained by each principal component:\n{var_exp}")

# 7. 计算累计解释的方差比例
cum_var_exp = np.cumsum(var_exp)
print(f"Cumulative proportion of variance explained:\n{cum_var_exp}")

# 7. 绘制 Scree 图和累计解释方差图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Scree 图
ax[0].plot(var_exp, marker="o")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Proportion of explained variance")
ax[0].set_title("Scree plot")

# 累计解释方差图
ax[1].plot(np.cumsum(var_exp), marker="o")
ax[1].set_xlabel("Principal component")
ax[1].set_ylabel("Cumulative sum of explained variance")
ax[1].set_title("Cumulative scree plot")

plt.tight_layout()
plt.show()


# 选取前5个, Cumulative proportion达到了 0.83


# Regression on selected components
# 选取5个
k = 5
principal_components = eig_vecs[:, :k]
X_new = np.dot(df_scaled, principal_components)


print(X_new)

X_new = sm.add_constant(X_new)
results = sm.OLS(y, X_new).fit()
print(results.summary())

# Remove x5  x5的p值为0.682，较大，去除
# 移除 x5 列，即第5个主成分，注意x5是第6列(索引为5)，使用 np.delete
X_new_removed = np.delete(X_new, 5, axis=1)

# 重新运行线性回归分析
results_removed = sm.OLS(y, X_new_removed).fit()
print(results_removed.summary())

# Remove x4, x4的p值为0.075 较大，去除
X_new_removed = np.delete(X_new_removed, 4, axis=1)

# 重新运行线性回归分析
results_removed = sm.OLS(y, X_new_removed).fit()
print(results_removed.summary())

# 因此，最后可以保留前三个主成分

# pca
pca = PCA()
pca.fit_transform(df_scaled)
print(pca.components_)  # Eigenvectors
# 最后得到的三个主成分应该是print(pca.components_)的前三列   PPT: Course4_Updated 31页
# 写出 Regression equation    PPT:Course4_Updated 41页
result = pca.components_[:, :3]
print(result)