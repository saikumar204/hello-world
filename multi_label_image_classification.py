import numpy as np
import h5py
import cv2
import scipy.io as sio
import os
from skimage import io
from skimage.transform import resize
-----------------------------------------------------------------------------------------------
#loading data images nad corrsponding target path 
target_path = "/content/sample_data/miml_data.mat"    # target labels  
image_path = "/content/sai1/mutli_label_datset/"     # images 

y = sio.loadmat(target_path)
y = y['targets']
print(y)
y = y.transpose()
print(y)
y = np.array([[elem if elem == 1 else 0 for elem in row]for row in y])
print(y)
x = []

for i in range(1,2001):
    #print("reading image:"+str(i) + ".jpg")
    img = image_path + "/" + str(i) + ".jpg"
    #print(img)
    img = cv2.imread(img)
    img = cv2.resize(img,(100,100))
    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    img = img.transpose((2,0,1))
    #print(img)
    # img = io.imread(img)
    # img = resize(img,(100,100))
    # img = img.transpose()
    x.append(img)
    
    
x = np.array(x)
f = h5py.File("dataset.h5")
f['x'] = x
f['y'] = y
print(f['x'])
print(f['y'] )
f.close()

[[ 1  1  1 ... -1 -1 -1]
 [-1 -1 -1 ... -1 -1 -1]
 [-1 -1 -1 ... -1 -1 -1]
 [-1 -1 -1 ... -1 -1 -1]
 [-1 -1 -1 ...  1  1  1]]
[[ 1 -1 -1 -1 -1]
 [ 1 -1 -1 -1 -1]
 [ 1 -1 -1 -1 -1]
 ...
 [-1 -1 -1 -1  1]
 [-1 -1 -1 -1  1]
 [-1 -1 -1 -1  1]]
[[1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 ...
 [0 0 0 0 1]
 [0 0 0 0 1]
 [0 0 0 0 1]]
<HDF5 dataset "x": shape (2000, 3, 100, 100), type "|u1">
<HDF5 dataset "y": shape (2000, 5), type "<i8">
--------------------------------------------------------------------------------------------------
# load h5 file and train the model 

import numpy as np
#from getdata import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
#K.set_image_dim_ordering('th')

import h5py
from sklearn.model_selection import train_test_split

def load():
  f = h5py.File("/content/sai1/dataset.h5")
  x = f['x'].value
  y = f['y'].value
   # print(x[0])
   # print(y[0])
  f.close()
  x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
  return x_train, x_test, y_train, y_test
  
  
x_train, x_test, y_train, y_test = load()
#print(x_train[0], y_train[0])

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255

#print(x_train[0],y_train[0])

model = Sequential()

#print(model)

model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
model.add(Activation('relu'))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))

model.add(Convolution2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.load_weights("weights.hdf5")
model.summary()

check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

model.fit(x_train, y_train, batch_size=32, nb_epoch=20,callbacks=[check],validation_data=(x_test,y_test))
----------------------------------------------------------------------------------------------------------------
# predict the model for test images 
model.summary()

#model = Sequential()

out = model.predict_proba(x_test)
out = np.array(out)
print(out[0])
print(out.shape[1])
acc = []
accuracies = []
threshold = np.arange(0.1,0.9,0.1)
best_threshold = np.zeros(out.shape[1])
print(best_threshold)

for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(y_test[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []
print(best_threshold)
y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
#print("algo",y_pred)

#print("gd",y_test)
print("y_pred",y_pred[0])
print("y_test",y_test[0])

hamming_loss(y_test,y_pred) 

total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 5])
print(total_correctly_predicted)

from IPython.display import Image 
Image(filename='/content/sai1/mutli_label_datset/500.jpg')

import cv2

img = cv2.imread("/content/sai1/mutli_label_datset/500.jpg")

print(img.shape)

img = cv2.resize(img,(100,100))
print(img.shape)


img = img.transpose((2,0,1))

img = img.astype('float32')

img = img/255

img = np.expand_dims(img,axis=0)

print(img.shape)

pred = model.predict(img)

print(pred)


y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])


print(y_pred)

classes = ['desert','mountains','sea','sunset','trees']

[classes[i] for i in range(5) if y_pred[i]==1 ] 
----------------------------------------------------------------------------------------------------------------------------

# o/p predictions
[0.0284088  0.04938084 0.9044822  0.01526973 0.06737295]
5
[0. 0. 0. 0. 0.]
[0.6 0.3 0.3 0.5 0.4]
y_pred [0 0 1 0 0]
y_test [0 0 1 0 0]
206
(256, 384, 3)
(100, 100, 3)
(1, 3, 100, 100)
[[0.05968812 0.7637312  0.13388446 0.01373658 0.3166564 ]]
[0 1 0 0 0]
['mountains']

