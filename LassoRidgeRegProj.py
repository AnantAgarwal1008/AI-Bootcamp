#ques1--implement polynomial regression and visualize the fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
import seaborn as sns


#load the california housing dataset
data=fetch_california_housing(as_frame=True)
df=data.frame

#select feature(median income) and target ( median house value)

print(df.head()) #to check how the data looks
x=df[['MedInc']]
y=df[['MedHouseVal']]

#transform feature to polynomial features
poly=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly.fit_transform(x)

#fit the polynomial regression model
model=LinearRegression()
model.fit(x_poly,y)

#make predictions:
y_pred=model.predict(x_poly)

#plot actual vs predicted data
plt.figure(figsize=(10,6))
plt.scatter(x,y,color="red",label="actual",alpha=0.5)
plt.scatter(x,y_pred,color="blue",label="predictted",alpha=0.5)
plt.title("polynomial regression")
plt.xlabel("median house income")
plt.ylabel("median house value")
plt.legend()
plt.show()

#evaluATe model performance
mse=mean_squared_error(y,y_pred)
print("mean squared error : ",mse)


#ques2--use lasso and ridge regression

#split data into training and testing
x_train,x_test,y_train,y_test=train_test_split(x_poly,y,test_size=0.2,random_state=42)

#ridge regression
ridge_model=Ridge(alpha=1)
ridge_model.fit(x_train,y_train)
ridge_predictions=ridge_model.predict(x_test)

#ridge regression
lasso_model=Lasso(alpha=1)
lasso_model.fit(x_train,y_train)
lasso_predictions=lasso_model.predict(x_test)

#evaluate ridge regression
ridge_mse=mean_squared_error(y_test,ridge_predictions)
print("ridge regression mse : ",ridge_mse)

#evaluate lasso regression
lasso_mse=mean_squared_error(y_test,lasso_predictions)
print("lasso regression mse : ",ridge_mse)

#visualize ridge vs lasso predictions
plt.figure(figsize=(10,6))
plt.scatter(x_test[:,0],y_test,color="red",label="actual",alpha=0.5)
plt.scatter(x_test[:,0],ridge_predictions,color="blue",label="ridge predictted",alpha=0.5)
plt.scatter(x_test[:,0],lasso_predictions,color="orange",label="lasso predictted",alpha=0.5)
plt.title("ridge vs lasso regression")
plt.xlabel("median house income")
plt.ylabel("median house value")
plt.legend()
plt.show()



#ques3--include more features and observe the impact on model performance


x1=df[['MedInc','HouseAge']]
y1=df[['MedHouseVal']]

#transform feature to polynomial features
poly=PolynomialFeatures(degree=2,include_bias=False)
x_poly1=poly.fit_transform(x)

#fit the polynomial regression model
model=LinearRegression()
model.fit(x_poly1,y)

#make predictions:
y_pred=model.predict(x_poly1)
#visualize relationships
sns.pairplot(df, x_vars=["MedInc","HouseAge"],y_vars=["MedHouseVal"],height=5,aspect=0.8,kind="scatter")
plt.title("median income and house age relationship")
plt.show()