#import python dependancies and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#initialise column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#import dataset into a dataframe
# dataset = pd.read_csv(url, names=names)
dataset = pd.read_csv("iris.data", names = names);

# print(dataset.head());
#print(dataset.describe())

#get required columns for x and y, splitting data into attributes and labels
x = dataset.iloc[:,:-1].values #all data columns
y = dataset.iloc[:,4].values   #categorial variable column

#split dataset into training and test
#model tested on un-seen data, x = data, y = classifier, using 50/50 train/test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

#Displaying Scatterplot of Iris feature data
plt.xlabel('Features')
plt.ylabel('Class')

pltX = dataset.loc[:,'sepal-length']
pltY = dataset.loc[:,'Class']
plt.scatter(pltX, pltY, color='blue', label='sepal-length')

pltX = dataset.loc[:, 'sepal-width']
pltY = dataset.loc[:,'Class']
plt.scatter(pltX, pltY, color='green', label='sepal-width')

pltX = dataset.loc[:, 'petal-length']
pltY = dataset.loc[:,'Class']
plt.scatter(pltX, pltY, color='red', label='petal-length')

pltX = dataset.loc[:, 'petal-width']
pltY = dataset.loc[:,'Class']
plt.scatter(pltX, pltY, color='black', label='petal-width')

plt.legend(loc=4, prop={'size':8})
plt.title('Scatterplot of Iris Features')
# plt.show()

#Creating logistic regression model and fitting to training data
model = LogisticRegression()
model.fit(X_train, y_train)
log_pred = model.predict(X_test)

#Printing classification accuracy results 
print(log_pred)
print("Accuracy: " + str(metrics.accuracy_score(y_test, log_pred)))
print(classification_report(y_test, log_pred))
