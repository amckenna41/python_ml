#import python dependancies and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#initialise column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#import dataset into a dataframe
# dataset = pd.read_csv(url, names=names)
dataset = pd.read_csv("iris.data", names = names);

#print(dataset.head());
#print(dataset.describe())

# Setting values for species categorial variable
dataset.iloc[:50,4] = 0
dataset.iloc[50:101,4] = 1
dataset.iloc[101:150,4] = 2

#get required columns for x and y, splitting data into attributes and labels
x = dataset.iloc[:,:-1].values #all data columns
y = dataset.iloc[:,4].values   #categorial variable column

#split dataset into training and test
#model tested on un-seen data, x = data, y = classifier, using 50/50 train/test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.5)

#scaling and normalising features so each feature contributes proportionally
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# array that holds all tree values used to create models
num_trees = [10,25,50,100,150,200,250,500,1000]

#create RandomForest model and calculate prediction accuracy
for i in range(0, len(num_trees)):
    model = RandomForestClassifier(n_estimators=num_trees[i], bootstrap =True, max_features='sqrt')
    #model = RandomForestClassifier(n_estimators=num_trees[i])
    model.fit(X_train, y_train)
    rf_pred = model.predict(X_test)
    print("Accuracy with " + str(num_trees[i]) + ' trees is ' + str(metrics.accuracy_score(y_test, rf_pred)))

#Creating model and calculating ROC Value

# model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
# model.fit(X_train, y_train)
# rf_pred = model.predict(X_test)
# rf_prob = model.predict_proba(X_test)[:,1]
# roc_val = roc_auc_score(y_test, rf_prob)
# print('Accuracy: ', metrics.accuracy_score(y_test, rf_pred))
