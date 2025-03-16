#build, tune, and optimize a machine learning model using a structured process and evaluate its performance comprehensively


import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np

#load datset for telcocustomer churn
df = pd.read_csv('C:/Users/user/Desktop/TelcoChurn.csv')

#display dataset info
print(df.info())
print("\n class distribution: \n",df['Churn'].value_counts())
print(df.head())

#handle missing values
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce') #convert to numeric and replace errors with NaN
df.fillna({'TotalCharges':df['TotalCharges'].median()},inplace=True) #fill NaN with median

#encode categorical variables
le=LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column!='Churn':
        df[column]=le.fit_transform(df[column])

#encode target variable
df['Churn']=le.fit_transform(df['Churn'])

#scale numerical features
scaler=StandardScaler()
numerical_features=['tenure','MonthlyCharges','TotalCharges']
df[numerical_features]=scaler.fit_transform(df[numerical_features])

print(df.head())

#features and target
x=df.drop(columns=['Churn'])
y=df['Churn']

#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train the model
rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(x_train,y_train)

#make predictions and evaluate
y_pred=rf_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print("Initial model accuracy: ",accuracy)
print("\n classification report: ",classification_report(y_test,y_pred))

#define parameter grid
param_grid = {'n_estimators': np.arange(50,200,10),'max_depth': np.arange(5,15,5),'min_samples_split': [2,5,10],'min_samples_leaf':[1,2,4]}

#initialize randomizedsearchCV
random_search=RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                 param_distributions=param_grid,
                                 n_iter=20,
                                 cv=5,
                                 scoring='accuracy',
                                 n_jobs=-1,
                                 random_state=42)

#perform randomized search
random_search.fit(x_train,y_train)

#get best parameters
best_params=random_search.best_params_
print("Best hyperparameters randomizedsearchcv: ",best_params)

#get best model
best_model=random_search.best_estimator_

#predict and evaluate
y_pred=best_model.predict(x_test)
accuracy_tuned=accuracy_score(y_test,y_pred)

print("Tuned model accuracy: ",accuracy_tuned)
print("\n classification report: ",classification_report(y_test,y_pred))

#evaluate cross validation
cv_scores=cross_val_score(best_model,x,y,cv=5,scoring='accuracy',n_jobs=-1)

print("Cross validation accuracy scores : ",cv_scores)
print("Mean cross validation accuracy: ",cv_scores.mean())
