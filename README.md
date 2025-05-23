# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset using pandas.
2. Separate the features and target variable from the dataset.
3. Convert feature values to float type for processing.
4. Apply standard scaling to both the features and target variable.
5. Initialize the model parameters (theta) to zeros.
6. Add a column of ones to the feature matrix to account for the intercept.
7. Perform gradient descent by iteratively updating theta using the prediction error.
8. After training, scale the new input data using the same scaler.
9. Predict the output using the learned parameters and scale it back to the original range.
10. Print the final predicted value.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  # Perform gradient descent
  for _ in range(num_iters):
    # Calculate predictions
    predictions = (X).dot(theta).reshape(-1, 1)
    # Calculate errors
    errors = (predictions - y).reshape(-1,1)
    # Update theta using gradient descent
    theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

![image](https://github.com/user-attachments/assets/98708f61-c593-450e-9f3f-51fee85d9587)

![image](https://github.com/user-attachments/assets/e7ee642a-1944-44be-acd9-43c30a47067f)

![image](https://github.com/user-attachments/assets/60d1bef2-09bb-4233-bb26-a41b99ffab20)

![image](https://github.com/user-attachments/assets/10706c95-96e6-4380-893f-36ceff74dc73)

![image](https://github.com/user-attachments/assets/a43dc8da-bbd2-403f-a804-a21584aeb90d)

![image](https://github.com/user-attachments/assets/5397c415-d54a-44a9-9fcd-c5be318466d6)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
