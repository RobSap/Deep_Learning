#This program is an example problem from Deep Learning with Python, by Francois Challet. 

#Import
from keras.datasets import imdb
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():

    #load the data
    #(x_train, y_train), (x_test, y_test) = imdb.load_data()
    
    #Limit the  number of words to the top 10,000 most frequently occuring words in the training data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    #The y (labels) are 0 (negative) or 1 (positive) representing reviews.

    #This is a way to decode the review back into english
    
    #word_index = imdb.get_word_index()
    #reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    #The ? is for missing words
    #Offset by 3 for padding
    
    #decoded_review = ' '.join ([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
    #print(decoded_review)

    #Using one-hot encoder
    def vectorize_sequences(sequences, dimension=10000):    
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i,sequence]=1
        return results
    x_train = vectorize_sequences(x_train)
    x_test =  vectorize_sequences(x_test)

    #Vectorize the labels
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001), activation='relu',   input_shape= (10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    #model.add(layers.Dense(16, activation='relu',   input_shape= (10000,)))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(16, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))


    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])
    #model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

    #Make a validation set
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    #print(len(x_val))
    #print(len(x_val[0]))
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    #print(y_val)
    #epochs = 10
    epochs = 7
    history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val,y_val))
     
    history_dict = history.history
  
    loss_values = history_dict['loss']
    loss_values = np.array(loss_values)
   

    val_loss_values = history_dict['val_loss']
    val_loss_values = np.array(val_loss_values)
    
    epochs = range(1, len(loss_values) + 1) 

    plt.plot(epochs, loss_values, label='Training loss')
    plt.plot(epochs, val_loss_values, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #Generic approach
    #model.fit(x_train,y_train, epochs=4, batch_size=512)
    loss,accuracy = model.evaluate(x_test,y_test)
 
    print("loss: "+ str(loss))
    print("accuracy: "+ str(accuracy))

    print("Running LR")
    lrmodel =LogisticRegression()
    fitmodel =lrmodel.fit(x_train, y_train)
    results = fitmodel.predict(x_test)
  
    #print(y_test) 
    scores = accuracy_score(y_test, results)
    print("LR")
    print("accuracy " + str(scores))
    

if __name__== "__main__":
  main()

