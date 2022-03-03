from pyexpat import model
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes.csv')

#loading all the csv data to a panda data frame
print(data)

#first 5 row of the dataset
first = data.head()
print(first)

#last 5 row fo the dataset
last = data.tail()
print(last)

#shape or number of rows and columns
num = data.shape
print(num)

#getting info about the data
info = data.info()
print(info)

#checking for missing values
missing = data.isnull().sum()
print(missing)

#statical measure about the data
stat = data.describe()
print(stat)

#checking the distribution of target values
distribute = data['Outcome'].value_counts()
print(distribute)

#1--> defective hear
#0--> health heart
#splinting feature and targets
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']
print(X)
print(Y)

#sprinting data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#model training
#logical Regression
model = LogisticRegression(max_iter=234567890890000)

#training the logistic regression model with training data
model.fit(X_train, Y_train)

#model evaluation
#accuracy score

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('accuracy on training data :', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy of the testing data :', test_data_accuracy)

#building a predictive system
input_data = (4,120,50,20,100,50,0.2,40)
#change input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

Prediction =  model.predict(input_data_reshaped)
print(Prediction)
