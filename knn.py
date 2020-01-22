#import python dependancies and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#import iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#initialise column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#import dataset into a dataframe
dataset = pd.read_csv(url, names=names)

#print dataset
print(dataset)

#get required columns for x and y
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#split dataset into training and test to avoid overfitting
#model tested on un-seen data
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

#feature scaling to normalise data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#train algorithm and make predictions, using scikit-learn
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Evaluating the algorithm
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Comparing error rate with k value
error = []

#calculate error rate in model prediction on test data
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#plot error rate using matplotlib library 
plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o',
        markerfacecolor='blue', markersize=10)
plt.title('Error rate for K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error rate')
plt.show()
