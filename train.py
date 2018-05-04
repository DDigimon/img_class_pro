from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.utils import np_utils,conv_utils
from keras.callbacks import TensorBoard
import numpy as np
import time

time_string=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
print(time_string)

import os
train_infor='./datasets/train.txt'
train_path='./datasets/train/'
test_path='./datasets/test/'
model_save_path='./save_data/model_data/'
tensorboard_path='./save_data/tb_data/'+str(time_string)


width=600
height=150
num_class=100
batch_size=64
epochs=1
X_train=[]
Y_train=[]
pic_list=[]
pic_num=[]
print('train_num:',len(os.listdir(train_path)))
train_num=len(os.listdir(train_path))
valid_num=int(train_num*(7/10))



with open(train_infor,encoding='utf-8') as fin:
    for num,line in enumerate(fin.readlines()):
        line=line.split('\n')[0].split(' ')
        Y_train.append(int(line[1])-1)
        img=load_img(os.path.join(train_path+line[0]),target_size=(height,width))
        img=img_to_array(img)
        X_train.append(img)
X_train=np.array(X_train)
Y_train=np_utils.to_categorical(Y_train,num_class)
print('Read Data Done')
print(np.shape(Y_train),X_train.shape)


X_valid=X_train[valid_num:]
Y_valid=Y_train[valid_num:]
X_train=X_train[:valid_num]
Y_train=Y_train[:valid_num]


X_train_datagen=ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./225,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
X_train_datagen.fit(X_train)

X_valid_datagen=ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./225,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
X_valid_datagen.fit(X_valid)



model=Sequential()
model.add(Conv2D(32,3,3,input_shape=(height,width,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(X_train_datagen.flow(X_train,Y_train,batch_size=batch_size),
                    validation_data=X_valid_datagen.flow(X_valid,Y_valid),
                    epochs=epochs,
                    callbacks=[TensorBoard(log_dir=tensorboard_path)])
model.save(os.path.join(model_save_path,str(time_string))+'model.h5')
