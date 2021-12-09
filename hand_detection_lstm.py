from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import random
import os

random.seed(10)

cwd='data/'
classes={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}  #人為設定2類
class_1 = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}

train_images = []
train_labels = []
test_images = []
test_labels = []
i = 0
for index, name in enumerate(classes):  # f_names为所有图片地址，list
    arr = []
    print("start loading {}".format(name), index, name)
    i += 1
    c = 0
    class_path=cwd+name+'/'
    arr_2 = [None for i in range(len(os.listdir(class_path)))]
    arr_1 = [int(i.split('.')[0]) for i in os.listdir(class_path)]
    for img_name in os.listdir(class_path):
        arr_1.sort()
        num = int(img_name.split('.')[0])
        img_path=class_path+img_name
        img = image.load_img(img_path, target_size=(128, 128))  # 读取图片
        arr_img = np.asarray(img)
        arr_2[arr_1.index(num)] = arr_img

    arr_3 = [i for i in arr_2 if type(i).__module__ == np.__name__]
    arr_2 = np.asarray(arr_3)
    print(arr_2.shape)
    for j in arr_2:
        c += 1
        train_images.append(j) # 把图片数组加到一个列表里面
        train_labels.append(int(name))

print("finish loading training images")
train_images = np.asarray(train_images)
print(train_images.shape)

cwd_1 = 'test_data/'
for index, name in enumerate(class_1):    #import testing data
    arr = []
    print("start loading {}".format(name), index, name)
    i += 1
    c = 0
    class_path=cwd_1+name+'/'
    arr_2 = [None for i in range(len(os.listdir(class_path)))]
    arr_1 = [int(i.split('.')[0]) for i in os.listdir(class_path)]
    for img_name in os.listdir(class_path):
        arr_1.sort()
        num = int(img_name.split('.')[0])
        img_path=class_path+img_name
        img = image.load_img(img_path, target_size=(128, 128))  # 读取图片
        arr_img = np.asarray(img)
        arr_2[arr_1.index(num)] = arr_img

    arr_3 = [i for i in arr_2 if type(i).__module__ == np.__name__]
    arr_2 = np.asarray(arr_3)
    print(arr_2.shape)
    for j in arr_2:
        c += 1
        test_images.append(j) # 把图片数组加到一个列表里面
        test_labels.append(int(name))

print("finish loading testing images")
test_images = np.asarray(test_images)
print(test_images.shape)

train_images_1 = []
train_labels_1 = []
y = train_labels[0]
c = 0
for i in range(len(train_images)):    #包裝訓練資料
    if(train_labels[i] == y):
        c+= 1
    else:
        c = 1
    if(c == 20):
        train_images_1.append([train_images[x] for x in range(i-19, i+1)])
        train_labels_1.append(train_labels[i])
        c = 0
    if(type(train_labels[i]) is not int):
        print('not_int', end = ';')
    y = train_labels[i]
train_images = train_images_1
train_labels = train_labels_1
print(len(train_images))


train = list(zip(train_images, train_labels))   #打亂訓練資料
random.shuffle(train)
train_images, train_labels = zip(*train)
train_images, train_labels = list(train_images), list(train_labels)

#print(test_labels)
test_images_1 = []
test_labels_1 = []
y = test_labels[0]
c = 0
for i in range(len(test_images)):    #包裝訓練資料
    if(test_labels[i] == y):
        c+= 1
    else:
        c = 1
    if(c == 10):
        test_images_1.append([test_images[x] for x in range(i-9, i+1)])
        test_labels_1.append(test_labels[i])
        c = 0
    y = test_labels[i]
test_labels = test_labels_1
test_images = test_images_1

print(test_labels)

test = list(zip(test_images, test_labels))    #打亂測試資料
random.shuffle(test)
test_images, test_labels = zip(*test)
test_images, test_labels = list(test_images), list(test_labels)

for i in range(len(test_images)):
    for j in range(len(test_images[i])):
        ax = plt.subplot(1, len(test_images[i]), j + 1)
        plt.imshow(test_images[i][j], cmap = 'binary')
    print(test_labels[i], end = ' ')
    plt.show()

test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels)
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels)

test_images = np.asarray(test_images)
train_images = np.asarray(train_images)

print(train_labels.shape)
print(train_images.shape)


# 建立卷積神經網路
model = Sequential()
model.add(TimeDistributed(Conv2D( filters = 32, kernel_size = ( 3, 3 ), 
	          activation = 'relu', 
	         padding = 'same' ), input_shape = (None, 128, 128, 3)))
model.add( TimeDistributed(BatchNormalization()))
model.add( TimeDistributed(MaxPooling2D( pool_size = ( 2, 2 ))))
model.add( Dropout(0.5))
model.add( TimeDistributed(Conv2D( filters = 64, kernel_size = ( 3, 3 ), 
	                 activation = 'relu', padding = 'same' )))
model.add( TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D( pool_size = ( 2, 2 ) )))	
model.add( Dropout(0.5))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(500, name="lstm_layer_rgb"))  #131072
model.add(Dense(13, activation = 'softmax'))
model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', 
	             metrics = ['accuracy'] )
print( model.summary() )
model.fit( train_images, train_labels, epochs = 20, batch_size = 1)

model.save("model_lstm_13.h5")

# 測試階段
test_loss, test_acc = model.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )

predictions = model.predict_classes(test_images)
print(predictions, test_labels)
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    ax.imshow(test_images[i][0], cmap = 'binary')
    ax.set_title('label=' + str(np.where(test_labels[i] == 1)[0]) + '\npredict=' + str(predictions[i]))
    print(predictions[i], str(np.where(test_labels[i] == 1)[0]))
plt.show()