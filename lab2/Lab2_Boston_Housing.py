import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data set in Python
df = pd.read_excel('BostonHousing.xlsx', sheet_name='Feuil1')
# print(df.head())
selected_df = df.iloc[:, 1:12]

# Pairwise scatter plots
#  the relationships and distributions among the selected features
#  (crime rate, non-retail business proportion, nitric oxides concentration, and median home value)
sns.pairplot(df[['crim', 'indus', 'nox', 'medv']], height=2)
plt.show()

# Scatter plots
# We can observe that the concentration of nitric oxide is very low in the non-retail commercial area in each town
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df['indus'], y=df['nox'])
plt.title('Scatter Plot of NOX vs INDUS')
plt.xlabel('INDUS')
plt.ylabel('NOX')
plt.show()

# Heatmap
# Correlation coefficient between each pair of selected variables
plt.figure(figsize=(10, 8))
sns.heatmap(selected_df.corr(), annot=True, linewidths=0.5)
plt.title('Heatmap of Selected Columns')
plt.show()