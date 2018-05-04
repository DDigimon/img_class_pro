from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import os

test_path='./datasets/test/'
result_file='test.txt'
model_path='./save_data/model_data/2018-05-04-11-43-25model.h5'

width=600
height=150

model=load_model(model_path)


with open(result_file,'w',encoding='utf-8') as fin:
    for file in os.listdir(test_path):
        img=load_img(test_path+file,target_size=(height,width))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        preds=model.predict_classes(img)
        fin.write(file+' '+str(preds[0])+'\n')
