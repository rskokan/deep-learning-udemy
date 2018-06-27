# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Part I: Data preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the DUmmy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## Part II: Build the ANN
# It will have the Rectifier activation func. for the input and hidden layers and Sigmoid for the output one

import keras
from keras.models import Sequential
from keras.layers import Dense

# Init the ANN; it will be a classifier
classifier = Sequential()

# Add the input layer plus the first hidden one
# The first hidden layer will have avg(in_layer, out_layer) = avg(11, 1) = 6
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))

# 2nd hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Output layer; 1 output node: the client leaves the bank or not; Sigmoid activation func so that we know the probability of the client leaving
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
#   the 'adam' optimizer is a stochastic gradient descent algo
#   the 'binary_crossentropy' loss func is Logarithmic loss, suitable for the Sigmoid activation func of the output layer with binary out
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fit the ANN to the training data a.k.a. train the model!
classifier.fit(X_train, y_train, batch_size=10, epochs=100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Convert the probability 0..1 to True or False
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


correct_predictions = cm[0, 0] + cm[1, 1]
accuracy = correct_predictions / len(X_test)
print('The prediction accuracy is {}%'.format(accuracy * 100))
