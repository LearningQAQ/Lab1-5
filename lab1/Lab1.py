import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 1.Load the dataset
df = pd.read_excel('Lab1_Buffalo_Cleaned.xlsx', sheet_name='Lab1_Buffalo_Cleaned')

# 2.Exclude the 'Total' row in different years because we only want to analyze data for individual regions
df_without_total = df[df['Region'] != 'Total']

# 3.Present descriptive statistics for female, male and total separately
descriptive_statistics = df_without_total[['Female', 'Male', 'Total']].describe()

# Print descriptive statistics
print(descriptive_statistics)


# 4.Boxplot of Female, Male, and Total separately
plt.figure(figsize=(9, 12))
sns.boxplot(data=df_without_total[['Female', 'Male', 'Total']])
plt.ylabel('The Number of Cattles')
plt.title('Boxplot of Female, Male, and Total separately')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# 5.Boxplot of Total Cattles for Each Region
plt.figure(figsize=(9, 12))
sns.boxplot(x='Region', y='Total', data=df_without_total)
plt.title('Boxplot of Total Cattles for Each Region')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
