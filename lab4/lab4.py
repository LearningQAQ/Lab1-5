import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('data_Labo4.csv', sep=';')
# print(df.head())


# # Part 1
# X = df[['width']]  # 使用壳宽作为自变量
# X = sm.add_constant(X)
#
# y = df['y']  # 使用y作为因变量
#
# # 拟合逻辑回归模型 logit
# logit_model = sm.Logit(y, X).fit()
# print(logit_model.summary())
#
# # glm
# glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
# print(glm_model.summary())
#
#
# # Extract the coefficients
# coefficients = logit_model.params
#
# # Calculate the log odds for a 25 cm female crab
# width = 25
# log_odds = coefficients.iloc[0] + coefficients.iloc[1] * width
# print(f"Log odds of a 25 cm female having satellites: {log_odds}")  # 0.07994695506766725
#
# # Calculate the probability
# probability = 1 / (1 + np.exp(-log_odds))
# print(f"Probability of a 25 cm female having satellites: {probability}") # 0.5199761001038238


# Part 2
X = df[['weight']]
X = sm.add_constant(X)
y = df['y']

logit_model_2 = sm.Logit(y, X).fit()

print(logit_model_2.summary())
coefficients_2 = logit_model_2.params

print(f"Regression equation: log(p / (1 - p)) = {coefficients_2.iloc[0]} + {coefficients_2.iloc[1]} * weight")
weight = 2000
log_odds_2 = coefficients_2.iloc[0] + coefficients_2.iloc[1] * weight


probability = 1 / (1 + np.exp(-log_odds_2))
print(f"Probability of a female crab weighing {weight} grams having satellites: {probability}") # 0.4838962679548775

