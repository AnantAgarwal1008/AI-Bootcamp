import pandas as pd
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report,roc_auc_score

#load dataset
df=pd.read_csv("C:\\Users\\user\\Desktop\\AI Bootcamp\\AI\\MachineLearning\\TelcoChurn.csv")

#display datsaet info and preview
print("Dataset Info")
print(df.info())
print("class distribution")
print(df['Churn'].value_counts())
print("Dataset Preview")
print(df.head())

#handle missing values
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna({'TotalCharges':df['TotalCharges'].median()},inplace=True)

#encode categorical variables
encoder=LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col!='Churn':
        df[col]=encoder.fit_transform(df[col])

#encode target variable
df['Churn']=encoder.fit_transform(df['Churn'])

#scale numerical features

scaler=StandardScaler()
numerical_features=['tenure','MonthlyCharges','TotalCharges']
df[numerical_features]=scaler.fit_transform(df[numerical_features])

#feature and target
X=df.drop('Churn',axis=1)
y=df['Churn']

#split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#apply SMOTE
smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

#display class distribution after SMOTE
print("Class distribution after SMOTE")
print(y_train_smote.value_counts())

#train Random Forest Classifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_smote,y_train_smote)
y_pred_rf=rf.predict(X_test)
roc_auc_score_rf=roc_auc_score(y_test,rf.predict_proba(X_test)[:,1])

#train XGBoost Classifier
xgb=XGBClassifier(random_state=42,eval_metric='logloss')
xgb.fit(X_train_smote,y_train_smote)
y_pred_xgb=xgb.predict(X_test)
roc_auc_score_xgb=roc_auc_score(y_test,xgb.predict_proba(X_test)[:,1])

#train LightGBM Classifier
lgb=LGBMClassifier(random_state=42,verbose=-1)
lgb.fit(X_train_smote,y_train_smote)
y_pred_lgb=lgb.predict(X_test)
roc_auc_score_lgb=roc_auc_score(y_test,lgb.predict_proba(X_test)[:,1])

#display classification report
print("Random Forest Classifier")
print(classification_report(y_test,y_pred_rf))
print("XGBoost Classifier")
print(classification_report(y_test,y_pred_xgb))
print("LightGBM Classifier")
print(classification_report(y_test,y_pred_lgb))

#display ROC AUC score
print("ROC AUC Score")
print("Random Forest Classifier:",roc_auc_score_rf)
print("XGBoost Classifier:",roc_auc_score_xgb)
print("LightGBM Classifier:",roc_auc_score_lgb)