#This is the MNist (National Institute of Standards and Technology ) data set that comes with Keras
#10,000 test images
#60,000 training images

#The hello world of deep learning

#Import
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import regularizers
import sys

def main():

    #load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    
    #print(x_train.shape)
    #sys.exit(1)
    #Display an image before changing dimensions
    '''
    digit = x_train[0]
    import matplotlib.pyplot as plt
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    '''
   
    #Create the model
    model = models.Sequential()
 
    #Create the hidden layer
    # 98%
    #model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),activation = 'relu', input_shape = (28*28,)))
    # ??
    model.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
    

    #This is the output layer, need 10 for 10 classificationsi, returns an array of 10
    model.add(layers.Dense(10,activation='softmax'))
    
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    #Change the shape for single dim output
    #Change values from uint8 (o to 255) to float32 (0 to 1)
    x_train = x_train.reshape((60000,28*28))
    x_train = x_train.astype('float32')/255    
    x_test = x_test.reshape((10000, 28*28))
    x_test = x_test.astype('float32')/255

    #Turn the output y into an array of [0,1]'s to to be categorical.
    #ex[0,0,0,0,0,0,0,0,1] - would indicate the value is a 9
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    
    #Fit/Train the model 
    model.fit(x_train, y_train, epochs = 5, batch_size=128)
    #model.fit(x_train, y_train, epochs = 5, batch_size=128)
 
    #See how accurate the training model is against a test set
    loss, accuracy = model.evaluate(x_test,y_test)
    print("Loss " + str(loss))
    print("Accuracy " + str(accuracy))


main()
