# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dharshni V M
RegisterNumber: 212223240029 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(2)
df.tail(4)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='violet')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
/*
```

## Output:

### DATA SET
![Data Set](Dataset-1.jpg)

### HEAD VALUES
![Head Values](Headvalues-1.jpg)

### TAIL VALUES
![Tail Values](Tailvalues-1.jpg)

### X VALUES
![X Values](X-1.jpg)

### Y VALUES
![Y Values](Y-1.jpg)

### PREDICTION VALUES
![Prediction Values](Ypred-1.jpg)

![prediction Values](Ytest-1.jpg)

### MSE,MAE and RMSE
![MSE,MAE,RMSE](values-1.jpg)

### TRAINING SET
![Training](Training-1.jpg)

### TESTING TEST
![Testing](Testing-1.jpg)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
