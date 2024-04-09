# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.

2. Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE,MAE and RMSE.

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
![Data Set](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/f0418cc4-a8b0-4ac8-8502-f3f76d0c959b)

### HEAD VALUES
![head values](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/ac9aee91-4d20-4b45-990b-3ee6a5dde196)

### TAIL VALUES
![Tail values](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/b9280e7e-85f5-46a9-964b-12e242cd0946)

### X VALUES
![X values](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/d87e447b-72a5-4dfe-97d9-6f6f88f2cce3)

### Y VALUES
![Y values](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/bc83164e-aff2-460e-a036-957c8a13352f)

### PREDICTION VALUES
![Y pred1](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/e8956985-c738-410e-adb2-6ad3e7769843)

![Y test1](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/fc3afccd-5131-4125-99ba-56a7790c9399)

### MSE,MAE and RMSE
![MSE,MAE,RMSE](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/825ff28f-a984-4964-b467-8a5ea460e5c6)

### TRAINING SET
![Training set](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/5c517f10-8a3a-40c8-8f7b-5afe00bc7701)

### TESTING TEST
![Testing set](https://github.com/Dharshni10/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145801097/54b77707-6a6f-4617-ac99-379c17511a0f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
