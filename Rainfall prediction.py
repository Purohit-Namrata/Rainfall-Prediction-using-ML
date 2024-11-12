import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Rainfall prediction in AUS/weatherAUS.csv")

X=dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
y=dataset.iloc[:,-1].values  #-1 for last col with all rows

#print(X)
#print(y)

y=y.reshape(-1,1)  #change 1D list to 2D list
print(y)

#Dealing with Invalid dataset

imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X=imputer.fit_transform(X)
y=imputer.fit_transform(y)

#print(X)
#print(y)

#Now, the dataset is almost clean
#encoding the data i.e. converting to numerical data

le1=LabelEncoder()
X[:,0]=le1.fit_transform(X[:,0])
le2=LabelEncoder()
X[:,4]=le2.fit_transform(X[:,4])
le3=LabelEncoder()
X[:,6]=le3.fit_transform(X[:,6])
le4=LabelEncoder()
X[:,7]=le4.fit_transform(X[:,7])
le5=LabelEncoder()
X[:,-1]=le5.fit_transform(X[:,-1])
le6=LabelEncoder()
y=le6.fit_transform(y)

#print(X)
#print(y)

#Feature scaling
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)  

#Split dataset into training set and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#print(X_train)
#print(y_train)

#Training model
model=RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X_train,y_train)

#print(model.score(X_train,y_train)) #Output: 0.99 i.e 99% of data is trained

print(y_test)
y_test=y_test.reshape(-1,1)
print(y_test)

y_pred=model.predict(X_test)
print(y_pred)

#Convert y_pred into yes or no

y_pred=le6.inverse_transform(y_pred)
print(y_pred)

y_test=le6.inverse_transform(y_test)


print("Accuracy score ",accuracy_score(y_test,y_pred))
 

