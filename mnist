import tensorflow as tf
import keras as K
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
#print(X_train[0])
#print(y_train[0])
#print(X_test[0])
#plt.imshow(X_train[0])
#X_train = X_train.reshape(60000,28,28,1)
#X_test = X_test.reshape(60000,28,28,1)

size2 = X_train.size
print(size2)

X_train = X_train.reshape(-1,28, 28, 1) 
size = X_train.size
print(size)
X_test = X_test.reshape(-1,28,28,1)


from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
size1 = y_train.size
print(size1)
y_train[0]
print(y_train[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
