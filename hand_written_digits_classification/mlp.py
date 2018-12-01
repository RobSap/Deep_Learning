#Base on keras-team/keras implementation of MLP
#https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np

seed = 9
np.random.seed(seed)

batch_size = 128
#num_classes = 10
epochs = 50

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape[0])
#print(x_train.shape[1])
#print(x_train.shape[2])
#print(x_train.shape[3])
'''
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
'''


#print(y_train)
#print(y_test)
#print(x_train)
#print(x_test)

#print(y_train)
#print(y_test)

#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)

#print(y_train)
#print(y_test)

x_train_new =[]
x_test_new =[]

'''
for z,each in enumerate(y_train):
    y_train_new.append([])
    #print("add image")
    for i,each2 in enumerate(each):
        y_train_new[z].append([])
        #print("add row")
        for j,each3 in enumerate(each2):
            try:
                y_train_new[z][i].append([])
                #print("add element into column")
                y_train_new[z][i][j]= int((int(each[i][j][0])+int(each[i][j][1])+int(each[i][j][2]))/3)/255
            except:
                print(each[i][j])
                print(z)
                print(i)
                print(j)
                print(y_train_new[z][i][j])

for z,each in enumerate(y_test):
    y_test_new.append([])
    #print("add image")
    for i,each2 in enumerate(each):
        y_test_new[z].append([])
        #print("add row")
        for j,each3 in enumerate(each2):
            try:
                y_test_new[z][i].append([])
                #print("add element into column")
                y_test_new[z][i][j]= int((int(each[i][j][0])+int(each[i][j][1])+int(each[i][j][2]))/3)/255
            except:
                print(each[i][j])
                print(z)
                print(i)
                print(j)
                print(y_test_new[z][i][j])
'''
for z,each in enumerate(x_train):
    x_train_new.append([])
    #print("add image")
    for i,each2 in enumerate(each):
        x_train_new[z].append([])
        #print("add row")
        for j,each3 in enumerate(each2):
            try:
                x_train_new[z][i].append([])
                #print("add element into column")
                x_train_new[z][i][j]= int((int(each[i][j][0])+int(each[i][j][1])+int(each[i][j][2]))/3)/255
            except:
                print(each[i][j])
                print(z)
                print(i)
                print(j)
                print(x_train_new[z][i][j])
x_test_new = []
for z,each in enumerate(x_test):
    x_test_new.append([])
    #print("add image")
    for i,each2 in enumerate(each):
        x_test_new[z].append([])
        #print("add row")
        for j,each3 in enumerate(each2):
            try: 
                x_test_new[z][i].append([])
                #print("add element into column")
                x_test_new[z][i][j]= int((int(each[i][j][0])+int(each[i][j][1])+int(each[i][j][2]))/3)/255
            except:
                print(each[i][j])
                print(z)
                print(i)
                print(j)
                print(x_test_new[z][i][j])


x_train = np.asarray(x_train_new)
x_test = np.asarray(x_test_new)

#y_train = np.asarray(y_train_new)
#y_test = np.asarray(y_test_new)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

'''
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
'''

x_train = x_train.reshape(-1, 1024)
x_test = x_test.reshape(-1, 1024)

num_classes = y_test.shape[1]
print("Output classes : " + str(num_classes))

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(num_classes, " output size")
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu', input_shape=(1024,)))#32, 32, 3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(128, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(56, activation='sigmoid'))
#model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

#model.summary()
#Adam
from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(),metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
