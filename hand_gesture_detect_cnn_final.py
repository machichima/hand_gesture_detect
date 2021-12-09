from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import SpatialDropout2D
import random
import os

cwd='data/'
classes={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}  #人為設定2類
#val_path = '/image/val/'

train_images = []
train_labels = []
test_images = []
test_labels = []

for index, name in enumerate(classes):  # f_names为所有图片地址，list
    c = 0
    class_path=cwd+name+'/'
    print("start loading {}".format(name), index, name)
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name
        #print(img_path)
        img = image.load_img(img_path, target_size=(128, 128))  # 读取图片
        #arr_img = image.img_to_array(img)  # 图片转换为数组 
        arr_img = np.asarray(img)
        #image = np.expand_dims(arr_img, axis=0)  #拓展维度
        #image=preprocess_input(image)
        train_images.append(arr_img) # 把图片数组加到一个列表里面
        train_labels.append([int(name)])
        #print("loading no.%s image."%index)
        #print(img_name)
        c += 1
print("finish loading training images")

cwd_test = 'test_data/'
test_images = []
test_labels = []
for index, name in enumerate(classes):  # f_names为所有图片地址，list
    class_path=cwd_test+name+'/'
    print("start loading {}".format(name), index, name)
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name
        img = image.load_img(img_path, target_size=(128, 128))  # 读取图片
        #arr_img = image.img_to_array(img)  # 图片转换为数组 
        arr_img = np.asarray(img)
        #image = np.expand_dims(arr_img, axis=0)  #拓展维度
        #image=preprocess_input(image)
        test_images.append(arr_img) # 把图片数组加到一个列表里面
        test_labels.append([int(name)])
print("finish loading training images")

#img_all = np.concatenate([x for x in train_images]) 

#train_images = imgs.reshape( ( 60000, 28 * 28 ) )   #60000
#train_images = train_images.astype( 'float32' ) / 255

train = list(zip(train_images, train_labels))
random.shuffle(train)
train_images, train_labels = zip(*train)
train_images, train_labels = list(train_images), list(train_labels)
#print(train_labels)


for i in range(13):
    plt.imshow(train_images[i], cmap = 'binary')
    plt.title('label=' + str(train_labels[i]))
    #print(predictions[i], str(np.where(test_labels[i] == 1)[0]))
    plt.show()
    
for i in range(13):
    plt.imshow(test_images[i], cmap = 'binary')
    plt.title('label=' + str(test_labels[i]))
    #print(predictions[i], str(np.where(test_labels[i] == 1)[0]))
    plt.show()

# test_images, test_labels = train_images[-20:], train_labels[-20:]
# train_images, train_labels = train_images[:-20], train_labels[:-20]
# print(len(train_images), len(train_labels))

test = list(zip(test_images, test_labels))
random.shuffle(test)
test_images, test_labels = zip(*test)
test_images, test_labels = list(test_images), list(test_labels)

test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels)
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels)

test_images = np.array(test_images)
train_images = np.array(train_images)



print(train_labels.shape)
print(train_images.shape)

# 建立卷積神經網路
network = Sequential( )
network.add( Conv2D( filters = 32, kernel_size = ( 3, 3 ), 
	         input_shape = (128, 128, 3), activation = 'relu', 
	         padding = 'same' ) )
network.add(SpatialDropout2D(0.5))
network.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )
network.add(SpatialDropout2D(0.5))
network.add( Conv2D( filters = 64, kernel_size = ( 3, 3 ), 
	                 activation = 'relu', padding = 'same' ) )
network.add(SpatialDropout2D(0.5))
network.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )	
network.add(SpatialDropout2D(0.5))
network.add( Flatten( ) )
network.add( Dense( 1024, activation = 'relu' ) )
network.add(Dropout(0.5))
network.add( Dense( 11, activation = 'softmax' ) )
network.compile( optimizer = 'adam', loss = 'categorical_crossentropy', 
	             metrics = ['accuracy'] )
print( network.summary() )
 
# 資料前處理
# train_images = train_images.astype( 'float32' ) / 255
# test_images = test_images.astype( 'float32' ) / 255
# train_labels = to_categorical( train_labels )
# test_labels = to_categorical( test_labels )

# 訓練階段

network.fit( train_images, train_labels, epochs = 20, batch_size = 100 )

network.save_weights("model.h5")
# 測試階段
test_loss, test_acc = network.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )


predictions = network.predict_classes(test_images)
print(predictions, end = ' ')

for i in range(11):
    plt.imshow(test_images[i], cmap = 'binary')
    plt.title('label=' + str(np.where(test_labels[i] == 1)[0]) + ' prediction:' + str(predictions[i]))
    #print(predictions[i], str(np.where(test_labels[i] == 1)[0]))
    plt.show()