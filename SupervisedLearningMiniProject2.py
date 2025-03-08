#classification-- Telco customer churn dataset

#task1-- perforn eda and preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#load datset for telcocustomer churn
df = pd.read_csv('C:/Users/user/Desktop/TelcoChurn.csv')

print(df.head())
df = df.drop(columns=['customerID']) #drop useless columns

# Handle missing values (numeric columns only)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode categorical variables
le = LabelEncoder() #here it will convert all categorical values to numerical using a loop
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())
#define features and target
x=df.drop(columns=['Churn']) #here it will select all columns except churn
y=df['Churn'] #here churn is selected as target

#scale features:
scaler=StandardScaler() #it scales all the data in same range thus helping ml to calculate easily with less error
x=scaler.fit_transform(x)

#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)

#check for best knn value
k_values = range(1, 21)  # Test k from 1 to 20
accuracy_scores = []

# Loop over k values
for k in k_values:
    # Initialize and train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = knn.predict(x_test)
    
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Find the best k
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Best k: {best_k}")
print(f"Accuracy for best k: {max(accuracy_scores)}")

#train best knn model
kmodel = KNeighborsClassifier(n_neighbors=best_k)
kmodel.fit(x_train, y_train)

#predict values
log_pred=model.predict(x_test)
knn_pred=kmodel.predict(x_test)

#make evaluation
print("logistic regression report : \n")
print(classification_report(y_test,log_pred))
print("knn report : \n")
print(classification_report(y_test,knn_pred))

#confusion matric for logistic regression
print("confusion matrix : \n",confusion_matrix(y_test,log_pred))