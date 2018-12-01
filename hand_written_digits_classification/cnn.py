import glob
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import re
import  PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras import optimizers
#from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import sys
#K.set_image_dim_ordering('th')


# Set the random see to reproduce results
seed = 9
numpy.random.seed(seed)

# (num_samples, 3, 32, 32)
# load data


image_list =glob.glob("CNN_Image_Classification/*.jpg")

X_train=[]
y_train=[]
y_train2=[]
X_test=[]
y_test=[]
y_test2=[]
labels = []
images = []

size = 32,32

train_size = 190
total_size = 270
test_size = 80

temp = []
for i in range(0,10):
   temp.append(0)

for i,each in enumerate(image_list):
    labels.append(each.replace("CNN_Image_Classification/","").replace(".jpg",""))
    labels[i] =int(re.sub('[^0-9]+','',labels[i]))
    #print(each)
    im = Image.open(each)
    im.thumbnail(size, Image.ANTIALIAS)
    im2arr = img_to_array(im)
    temp = []
    for z in range(0,10):
        temp.append(0)
 
    if labels[i] != 0:
        temp[labels[i]]=1
        y_train2.append(labels[i])
        y_test2.append(labels[i])

    X_train.append(im2arr)  
    y_train.append(temp)

    X_test.append(im2arr)
    y_test.append(temp)



# normalize inputs from 0-255 to 0.0-1.0

X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#num_classes = len(y_test[0])


#Make the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same',dim_ordering='tf'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adammax= optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
adam= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())


numpy.random.seed(seed)
model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_test), np.array(y_test)), epochs=epochs, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

