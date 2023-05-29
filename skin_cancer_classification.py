# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 23:03:53 2022

@author: Adel
"""

from keras.models import Sequential ,load_model
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization,Dropout,Activation
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import image_utils

## model prameters
filterSize = 3
dropRate = .2
input_dim = 64  #128
non_linear_fun = 'relu'
batch_size = 64


# The Model
model = Sequential()

#model.add(Conv2D(32,(filterSize,filterSize),input_shape=(64,64,3),activation='relu'))
model.add(Conv2D(64,(filterSize,filterSize),input_shape=(input_dim,input_dim,3),activation=non_linear_fun))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(dropRate))

model.add(Conv2D(64,(filterSize,filterSize),activation=non_linear_fun))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(dropRate))

# model.add(Conv2D(128,(filterSize,filterSize),activation=non_linear_fun))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2)))
# # model.add(Dropout(dropRate))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation(non_linear_fun))
model.add(BatchNormalization())
model.add(Dropout(dropRate))

model.add(Dense(128))
model.add(Activation(non_linear_fun))
model.add(BatchNormalization())
model.add(Dropout(dropRate))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Model Comiler
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.summary()


##############################################################################


# Data processing section(Data Augmentation)
## Creating generators 
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    shear_range=.2,
                                    zoom_range=.2,
                                    horizontal_flip=True,
                                    fill_mode='reflect')

dev_data_gen = ImageDataGenerator(rescale=1./255,
                                    shear_range=.2,
                                    zoom_range=.2,
                                    horizontal_flip=True,
                                    fill_mode='reflect')

test_data_gen = ImageDataGenerator(rescale=1./255)



## Feeding data to generators
import os
os.chdir(r'F:/')        # Make sure we are at the right directory

training_set = train_data_gen.flow_from_directory(
        r'skin_tumors_data\train',
        target_size=(input_dim, input_dim),
        batch_size=batch_size,
        class_mode='binary')

dev_set = dev_data_gen.flow_from_directory(
        r'skin_tumors_data/dev',
        target_size=(input_dim,input_dim),
        batch_size=batch_size,
        class_mode='binary')

test_set = test_data_gen.flow_from_directory(
        r'skin_tumors_data\test',
        target_size=(input_dim, input_dim),
        batch_size=batch_size,
        class_mode='binary',)

##############################################################################


# Fit the model and start training
history = model.fit(training_set,epochs=24,validation_data=dev_set,verbose=1)

# Evaluate the model
model.evaluate(test_set)
# model.save('skin_cancer_classfication_64_input_2conv_64_64_3dense_dropout=.2_25epoch.h5')
list_of_models = [['128_input_2conv_64_64_3dense_dropout=.2'            ,'acc:0.87','val_acc:0.82','test_acc:0.82'],
                  ['128_input_3conv_64_64_128_2dense_dropout=.2'        ,'acc:0.87','val_acc:0.82','test_val:0.81'],
                  ['128_input_3conv_32_64_128_3dense_dropout=.2'        ,'acc:0.87','val_acc:0.71','test_acc:0.68'],
                  ['128_input_3conv_32_64_128_3dense_no_regulaization'  ,'acc:0.90','val_acc:0.79','test_acc:0.76'],
                  ['128_input_2conv_64_64_3dense_dropout=.25'           ,'acc:0.87','val_acc:0.76','test_acc:0.75'],
                  ['64_input_2conv_64_64_3dense_dropout=.2_25epoch'     ,'acc:0.94','val_acc:0.87','test_acc;0.84']]

import pandas as pd
losses = pd.DataFrame(history.history)
losses[['loss','val_loss']].plot()
losses[['acc','val_acc']].plot()
model.metrics_names

## Testing 
#model = load_model('skin_cancer_classfication_fit.h5')
print(training_set.class_indices)       # {'benign': 0, 'malignant': 1}
print(dev_set.class_indices)            # {'benign': 0, 'malignant': 1}
print(test_set.class_indices)           # {'benign': 0, 'malignant': 1}

def predict_class(path,target_size,model):
    img = image_utils.load_img(path,target_size=(target_size,target_size))
    # img.getpixel
    plt.imshow(img)
    img = image_utils.img_to_array(img)
    img = img.reshape((1,) + img.shape)
    
    result = model.predict(img)
    if result[0][0] == 0:
        print('benign')
    elif result[0][0] == 1:
        print('malignant')

# predict_class(r'F:\skin_tumors_data\test\malignant\1173.jpg',input_dim,model)
