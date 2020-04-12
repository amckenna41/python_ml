#import python dependancies and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#initialise column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#import dataset into a dataframe
# dataset = pd.read_csv(url, names=names)
dataset = pd.read_csv("iris.data", names = names);

print(dataset.head());
#print(dataset)

#get required columns for x and y, splitting data into attributes and labels
x = dataset.iloc[:,:-1].values #all data columns
y = dataset.iloc[:,4].values   #categorial variable column

#split dataset into training and test
#model tested on un-seen data, x = data, y = classifier, using 50/50 train/test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

#scaling and normalising features so each feature contributes proportionally
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##Below code was used to find the best value for K, applying the algorithm for all values of k, 1-40

k = range(1,40)
class_error = []

for i in range(0, len(k)):
    print(k[i])
    correct = 0;
    n_neighbors = k[i]
    classifier = KNeighborsClassifier(n_neighbors=k[i])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    len_y = len(y_pred)
    class_error.append(np.mean(y_pred !=y_test))
    for i in range(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            correct = correct + 1
    print(str(correct)+ " predictions correct out of " + str(len_y) + ' with k = '+ str(n_neighbors));


#plotting classification error to help determine best value of k
plt.figure(figsize=(12,6))
plt.plot(range(1,40), class_error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#Low values of K and high values of k produce higher misclassification error
# #Ideal value of K = 15 - 25
# #train algorithm and make predictions
# n_neighbors = 20
# classifier = KNeighborsClassifier(n_neighbors)
# classifier.fit(X_train, y_train)
#
# #classify test data based on KNN model created
# y_pred = classifier.predict(X_test)
#
# # print(y_test)
# # print(y_pred)
#
# len_y = len(y_pred)
# correct = 0;
#
# for i in range(0, len(y_test)):
#     if y_test[i] == y_pred[i]:
#         correct = correct + 1;
#
# #Evaluating the algorithm with confusion matrix
# print(str(correct)+ " predictions correct out of " + str(len_y) + ' with k = '+ str(n_neighbors));
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
