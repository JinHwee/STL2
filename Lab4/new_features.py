import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

data0 = pd.read_csv('with_new_features.csv')
data = data0.drop(['Domain'], axis = 1).copy()

fn = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
            'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
            'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'subDomain', 'favicon']
cn =['Good','Bad']

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)

# Separating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12)

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

# instantiate the model 
dt = DecisionTreeClassifier(max_depth = 5)
# fit the model 
dt.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_tree = dt.predict(X_test)
y_train_tree = dt.predict(X_train)

#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)
print("(newFeature DecisionTree) Train accuracy: {:.3f}".format(acc_train_tree))
print("(newFeature DecisionTree) Test accuracy: {:.3f}".format(acc_test_tree))

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
tree.plot_tree(dt, feature_names = fn, class_names=cn, filled = True)
fig.savefig('new_feature_decision-tree.png')

################################################################# Multi Layer Perceptron #################################################################
model = Sequential()
model.add(Dense(1024, activation=('relu'), input_shape=(17,)))
model.add(Dense(512, activation=('relu')))
model.add(Dense(256, activation=('relu')))
model.add(Dense(128, activation=('relu')))
model.add(Dense(1, activation=('sigmoid')))

lr = 0.0001
loss = 'binary_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, momentum=0.9)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
training = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250)

# evaluate the model
_, train_acc = model.evaluate(X_train, y_train)
_, test_acc = model.evaluate(X_test, y_test)
print("(MLP) Train accuracy: {:.3f}".format(train_acc))
print("(MLP) Test accuracy: {:.3f}".format(test_acc))

################################################################ Random Forest ################################################################
randomforest = RandomForestClassifier(n_estimators=50, max_depth=5)
randomforest.fit(X_train, y_train)

train_predictions = randomforest.predict(X_train)
test_predictions = randomforest.predict(X_test)
print("(newFeature RandomForest) Train accuracy: {:.3f}".format(accuracy_score(y_train, train_predictions)))
print("(newFeature RandomForest) Test accuracy: {:.3f}".format(accuracy_score(y_test, test_predictions)))

################################################################# SVM #################################################################
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_train_predictions = svm.predict(X_train)
svm_test_predictions = svm.predict(X_test)
print("(newFeature SVM) Train accuracy: {:.3f}".format(accuracy_score(y_train, svm_train_predictions)))
print("(newFeature SVM) Test accuracy: {:.3f}".format(accuracy_score(y_test, svm_test_predictions)))


# Checking for overfitting and underfitting #
# fig, axes = plt.subplots(nrows = 1,ncols = 2)
# axes[0].plot(training.history['accuracy'])
# axes[0].plot(training.history['val_accuracy'])
# axes[0].set_title('model accuracy')
# axes[0].set_ylabel('accuracy')
# axes[0].set_xlabel('epoch')
# axes[0].legend(['train', 'val'], loc='upper left')

# axes[1].plot(training.history['loss'])
# axes[1].plot(training.history['val_loss'])
# axes[1].set_title('model loss')
# axes[1].set_ylabel('loss')
# axes[1].set_xlabel('epoch')
# axes[1].legend(['train', 'val'], loc='upper left')
# fig.savefig('newFeature accuracy-loss.png', dpi=400)