#make a project on to check if the upper class ticket people had a higher chances of survival in titanic
#task 1: perform data cleaning, aggregation and filtering
#task 2 : generate visualizations to illustrate key insights
#task3 : identify patterns or anomalies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#load titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

#Inspect data
print(df.info())
print(df.describe())

#Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

#remove duplicates
df=df.drop_duplicates()

#filter data: passengers in first class
first_class=df[df["Pclass"]==1]
print("First class Passengers:\n",first_class.head())


#Bar chart : Survival rate by class

survival_by_class=df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar",color="green")
plt.title("Survival rate by class")
plt.ylabel("survival rate")
plt.xlabel("passenger class")
plt.show()

#histogram : age distribution
sns.histplot(df["Age"],kde=True,bins=20,color="purple")
plt.title("age of distribution")
plt.xlabel("age")
plt.ylabel("frequency")
plt.show()

#scatter plot : age vs fare
plt.scatter(df["Age"],df["Fare"],alpha=0.5,color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

