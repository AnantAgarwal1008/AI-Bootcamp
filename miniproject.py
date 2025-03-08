#Project---Linear Regression from scratch
import numpy as np

#generate synthetic data
np.random.seed(42)
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)

#add bias term to feature matrix
x_b=np.c_[np.ones((100,1)),x]

#initialize parameters
theta=np.random.randn(2,1)
learning_rate=0.1
iteration=1000

#task1--implement the mathematical formula for linear regression
def predict(x,theta):
    return np.dot(x,theta)

#task2--use gradient descent to optimize the model parameters
def gradient_descent(x,y,theta,learning_rate,iteration):
    m=len(y)
    for _ in range(iteration):
        gradient=(1/m)*np.dot(x.T,(np.dot(x,theta)-y))
        theta-=learning_rate*gradient
    return theta


#task3--calculate evaluation metrics
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)
def r_squared(y_true,y_pred):
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    return 1-(ss_res/ss_tot)

#perform gradient descent
theta_optimized = gradient_descent(x_b,y,theta,learning_rate,iteration)

#prediction and evaluation
y_pred=predict(x_b,theta_optimized)
meanse=mse(y,y_pred)
r2=r_squared(y,y_pred)

print("optimized parameter : ",theta_optimized)
print("mse : ",meanse)
print("r2 : ",r2)
