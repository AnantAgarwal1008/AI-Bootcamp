import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#task1--we will do feature enggineering where we will standardize numerical feature encoding categorical feature and feature selection

#load titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

print(df.head())

#select releavant features
df=df[['Pclass','Sex','Age','Embarked','Fare','Survived']]

#handle the missing values
df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

#define feature and target
x=df.drop(columns=['Survived'])
y=df['Survived']

#apply feature scaling and encoding
preprocessor=ColumnTransformer(transformers=[('num',StandardScaler(),['Age','Fare']),('cat',OneHotEncoder(),['Pclass','Sex','Embarked'])])

x_preprocesses=preprocessor.fit_transform(x)

#task2--Train the model

#train and evaluate logistic regression model
log_model=LogisticRegression()
log_score=cross_val_score(log_model,x_preprocesses,y,cv=5,scoring='accuracy')
print("Logistic Regression mean accuracy : ",log_score.mean())

#train and evaluate random forest model
rf_model=RandomForestClassifier()
rf_score=cross_val_score(rf_model,x_preprocesses,y,cv=5,scoring='accuracy')
print("Random Forest mean accuracy : ",rf_score.mean())




#task3--apply grid search and hyperparameter tuning

#define hyperparameter grid
param_grid={'n_estimators':[50,100,200],
            'max_depth':[None,10,20],
            'min_samples_split':[2,5,10]
            }

#preform grid search
grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(x_preprocesses,y)

#display best hyperparameter and score
print("best hyperparameter:",grid_search.best_params_)
print("best accuracy : ",grid_search.best_score_)


