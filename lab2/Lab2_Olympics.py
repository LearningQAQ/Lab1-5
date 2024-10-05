import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1.Import the data set in Python
df = pd.read_excel('olympics.xlsx', sheet_name='olympics')
# print(df.to_string())

# 2.Rename columns
new_column_names = [
    'Country', 'Summer_Games', 'Gold', 'Silver',
    'Bronze', 'Total_Summer', 'Winter_Games', 'Gold', 'Silver',
    'Bronze', 'Total_Winter', 'Total_Games', 'Total_Gold',
    'Total_Silver', 'Total_Bronze', 'Combined total'
]
# 将前 15 列的名称更改为新的名称
df.rename(columns=dict(zip(df.columns[:16], new_column_names)), inplace=True)
print(df.head())

# 3.Missing values
df.fillna(0, inplace=True)
df.replace('No', 0, inplace=True)
# print(df.to_string())

# 4.Duplicates
no_duplicated = df.drop_duplicates() # Remove duplicate values
no_duplicated = no_duplicated.drop(0)
print(no_duplicated.head())
# print(no_duplicated.to_string())

# 5.Data types
# no_duplicated.info()
# print(no_duplicated.describe())

no_duplicated.iloc[:, 0] = no_duplicated.iloc[:, 0].astype(str) # 第一列
no_duplicated.iloc[:, 1:] = no_duplicated.iloc[:, 1:].astype(int) # 所有数字
# print(no_duplicated.to_string())

# 6.Outliers
problematic_rows = no_duplicated[(no_duplicated.iloc[:, 1:15] < 0).any(axis=1)]

# 输出有问题的行
print(problematic_rows.to_string())

# 经过观察，Finland (FIN)有两行不合理的数据,
# 在第一行的Finland (FIN)数据中，Bronze为-117，经计算，它的Bronze若为117即合理。
# 在第一行的Finland (FIN)数据中，Silver为-3，这一行不合理。
# 因此，保留第一行，将该值转为正数。去第二行这个不合理的行。
no_duplicated.iloc[40, 4] = abs(no_duplicated.iloc[40, 4])

no_outliers = no_duplicated.drop(43).reset_index(drop=True)

no_outliers.to_excel('new_olympics.xlsx', index=False)


# 先运行上面，把上面注释，再运行下面
# # 4.Plot any relevant graph that could help visualize this data set without scrolling through the file. Explain your choices for the graphs and the data they show.
# df = pd.read_excel('new_olympics.xlsx')
# print(df.head())
# df_excluding_last = df.iloc[:-1]  # 去除Total行
# top_10_combined_total = df_excluding_last.nlargest(10, 'Combined total') # Top 10 Countries by Combined Total
# # 1.奖牌总数箱线图
# plt.figure(figsize=(12, 9))
# sns.boxplot(data=df_excluding_last['Combined total'])
#
# plt.xlabel('Country')
# plt.ylabel('Combined Total Medals')
# plt.title('Box Plot of Combined Total Medals by Country')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
#
# # 2.Top Ten Medal Bar Table
# plt.figure(figsize=(12, 9))
# plt.bar(top_10_combined_total['Country'], top_10_combined_total['Combined total'], color='skyblue')
# plt.title('Top 10 Countries by Combined Total Medals (Excluding Last Entry)')
# plt.xlabel('Country')
# plt.ylabel('Combined Total Medals')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
#
# # 3. 前十名奖牌分布的堆叠柱状图
# medals = top_10_combined_total[['Country', 'Total_Gold', 'Total_Silver', 'Total_Bronze']]
# medals.set_index('Country').plot(kind='bar', stacked=True, figsize=(12, 9))
# plt.title('Distribution of Medals (Gold, Silver, Bronze) by Top 10 Countries')
# plt.xlabel('Country')
# plt.ylabel('Number of Medals')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 4. 绘制指定国家（例如中国）的奖牌饼图
# choose_country = 'China (CHN) [CHN]'  # 选择国家
# choose_data = [
#     df_excluding_last.loc[df_excluding_last['Country'] == choose_country, 'Total_Gold'].values[0],
#     df_excluding_last.loc[df_excluding_last['Country'] == choose_country, 'Total_Silver'].values[0],
#     df_excluding_last.loc[df_excluding_last['Country'] == choose_country, 'Total_Bronze'].values[0]
# ]
# labels = ['Gold', 'Silver', 'Bronze']
#
# plt.figure(figsize=(10, 10))
# plt.pie(choose_data, labels=labels, autopct='%1.2f%%', startangle=90)
# plt.title(f'Medal Distribution for {choose_country}')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()


