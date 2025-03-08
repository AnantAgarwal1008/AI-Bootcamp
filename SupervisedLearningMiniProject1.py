#Regression--California housing dataset


#task1-- perforn eda and preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#load dataset for california housing
data=fetch_california_housing(as_frame=True)
df=data.frame

#inspect dataset
print(df.info())
print(df.describe())

#visualize relationship
sns.pairplot(df,vars=['MedInc','AveRooms','HouseAge','MedHouseVal'])
plt.show()

#check for missing values
print("missing values : \n",df.isnull().sum())

#define target and feature
x=df[['MedInc','HouseAge','AveRooms']]
y=df['MedHouseVal']

#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train linear regression model
model=LinearRegression()
model.fit(x_train,y_train)

#make predictions
y_pred=model.predict(x_test)

#evaluate performnce
mse=mean_squared_error(y_test,y_pred)
print("mlinear regresssion mse : ",mse)
