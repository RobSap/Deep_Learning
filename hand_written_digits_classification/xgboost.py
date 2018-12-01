import glob
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import re
import  PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



X_train=[]
y_train=[]
X_test=[]
y_test=[]
labels = []
images = []

size = 32,32

train_size = 190
total_size = 270
test_size = 80

temp = []
for i in range(0,10):
   temp.append(0)

image_list =glob.glob("CNN_Image_Classification/*.jpg")

for i,each in enumerate(image_list):
    labels.append(each.replace("CNN_Image_Classification/","").replace(".jpg",""))
    labels[i] =int(re.sub('[^0-9]+','',labels[i]))
    print(each)
    im = Image.open(each)
    im.thumbnail(size, Image.ANTIALIAS)
    
    im2arr = img_to_array(im)
    temp = []
    

    X_train.append(im2arr)
    y_train.append(temp)

    X_test.append(im2arr)
    y_test.append(temp)

temp_list = []
for each in X_train: #list of images
    
    for i,each2 in enumerate(each): #height
        temp_list.append([])
        for j,each3 in enumerate(each2): #Width
           #print((each3[0] + each3[1] + each3[2]) /3)
           #try:
           #print(j)
           temp_list[i].append( ((each3[0] + each3[1] + each3[2]) /3))
           #except:
               #input("")
               #print(temp_list[i])

print(len(temp_list))
print(len(temp_list[0])) 

print("XGBoost")
model = GradientBoostingClassifier()
# of images
print(len(np.array(X_train)))
#height
print(len(np.array(X_train[0])))
#Width
print(len(np.array(X_train[0][0])))
print(len(np.array(y_train)))

#Wrong shape
model.fit(np.array(X_train), np.array(y_train))

#predictions = gbm.predict(X_test)
print("Running evaluations")
scores = gbm.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

