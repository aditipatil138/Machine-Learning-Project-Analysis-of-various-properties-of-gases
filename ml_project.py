# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

# improt dependencies
#importing peripherals
import pandas as pd
import numpy as np
#importing statistical library classes
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#importing plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import cm
"""Used in jupyter notebook/ Not used in spyder"""
#%matplotlib inlin


# # read data
df_main = pd.read_csv('C:/Users/CSUFTitan/PycharmProjects/Programs/Dataset.csv')
# # let's peek into the dataset
df_main.head()

# # describe the dataset
# df_main.describe()
# df_main.info()
# # calculate the correlation matrix
# corr = df_main.corr()
 
print("--------------------------------------------------")


corr_matrix = df_main.corr()
print("Correlation Matrix")
print("")
print(corr_matrix)
print("--------------------------------------------------")

# plot the correlation heatmap
#sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
#plt.scatter(df_main['Idx'], df_main['Sv'], alpha=0.5)
#plt.show(df_main['Idx'], df_main['Sv'], alpha=0.5)
#plt.scatter(df_main['Idx'], df_main['Th'], alpha=0.5)
#plt.show(df_main['Idx'], df_main['Th'], alpha=0.5)

#Pr having some error due same value ( 500 ) in dataset column


#Below are the scatter plots for properties.

fig, ax = plt.subplots()
my_scatter_plot = ax.scatter(df_main['Idx'], df_main['Th'])
plt.xlabel("Chemical Index")
plt.ylabel("Thermal Conductivity")


fig, ax = plt.subplots()
my_scatter_plot = ax.scatter(df_main['Idx'], df_main['Tm'])
plt.xlabel("Chemical Index")
plt.ylabel("Temperature")

fig, ax = plt.subplots()
my_scatter_plot = ax.scatter(df_main['Idx'], df_main['Sv'] )
plt.xlabel("Chemical Index")
plt.ylabel("Sound Velocity")

plt.show() 
    





# displays multiple figures ( without this it will only show heatmap and not scatter plot)


mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True



heatmap = sns.heatmap(corr_matrix, cmap = "YlOrRd",annot = True)


#add the column names as labels
# ax.set_yticklabels(corr_matrix.columns, rotation = 0)
# ax.set_xticklabels(corr_matrix.columns)
# sns.set_style({'xtick.bottom': True}, {'ytick.left': True})











columns_x = ['Th', 'Sv', 'Tm', ' Pr']
column_label = ['Idx']

#converting it into 2 dimension above ( col_x and col_lable)

X = df_main[columns_x]
y = df_main[column_label]

# Fitting the data into training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #taking testing set = 33% of total data and remaining percentage of data in x_train and Y_train
# print(X_test.shape)
reg = LinearRegression()
reg = reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2_score(y_test, y_pred)
# other way of calculating the R2 score
reg.score(X_test, y_test)
print("Regression Coefficients: ",reg.coef_)
print("--------------------------------------------------")
print("Regression Intercept: ",reg.intercept_)
print("--------------------------------------------------")
# split array in k(number of folds) sub arrays
X_folds = np.array_split(X_train, 3)
y_folds = np.array_split(y_train, 3)
scores = list()
models = list()
for k in range(3):
    reg = LinearRegression()

    # We use 'list' to copy, in order to 'pop' later on
    X_train_fold = list(X_folds)
    # pop out kth sub array for testing
    X_test_fold  = X_train_fold.pop(k)
    # concatenate remaining sub arrays for training
    X_train_fold = np.concatenate(X_train_fold)
    # same process for y
    y_train_fold = list(y_folds)
    y_test_fold  = y_train_fold.pop(k)
    y_train_fold = np.concatenate(y_train_fold)

    reg = reg.fit(X_train_fold, y_train_fold)
    scores.append(reg.score(X_test_fold, y_test_fold))
    models.append(reg)
    
print("Scores are:",scores)
print("--------------------------------------------------")
print("best linear score is",max(scores))
print("-----------Polynomial Regression------------------")
# polynomial model
for count, degree in enumerate([2, 3, 4, 5, 6]):
    print("Degree ",degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("R2 score: ",r2_score(y_test, y_pred))
    print("coefficiets: ",model.steps[1][1].coef_)
    print("bias: ",model.steps[1][1].intercept_)
    print("---------------------------------")
    print()
