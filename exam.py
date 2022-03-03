#Importing necessary libraries
import numpy as np
import pandas as pd

#Storing the values from the dataset in a variable
dataset = pd.read_csv("weatherAUS.csv")

#Dividing the values in the dataset into dependent(Y) and independent(X) variables and removing columns with non necessary values or too many null values i.e: 0,5,6
X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,18,19,20,21]].values
Y = dataset.iloc[:,-1].values

#Reshaping Y from a 1-dimensional(a[n]) array into a 2-dimensional(a[n][m]) array 
Y = Y.reshape(-1,1)

#Removing NA from the dataset and replacing it with the most frequent value in that column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

#Encoding non-numerical(i.e: W,WNW) values into numerical values(i.e: 1,2,3,4)
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
le5 = LabelEncoder()
X[:,-1] = le5.fit_transform(X[:,-1])
le6 = LabelEncoder()
Y = le6.fit_transform(Y)

#Feature scaling to minimize data scattering
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Dividing the dataset into 2 parts namely training data and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Training our model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state=0)
classifier.fit(X_train,Y_train)
classifier.score(X_train,Y_train)
Y_test = Y_test.reshape(-1,1)
Y_pred = classifier.predict(X_test)
Y_pred = le6.inverse_transform(Y_pred)
Y_test = le6.inverse_transform(Y_test)
Y_test = Y_test.reshape(-1,1)
Y_pred = Y_pred.reshape(-1,1)

#Concatenating our test and prediction result into a dataset
df = np.concatenate((Y_test,Y_pred),axis=1)
dataframe = pd.DataFrame(df,columns=['Rain Tomorrow','Rain Prediction'])

#Checking the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))

#Print .csv file
dataframe.to_csv("predictions.csv")
