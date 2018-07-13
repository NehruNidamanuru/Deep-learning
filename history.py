# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Wed May 16 18:53:36 2018)---
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
model = load_model('Downloads/model.h5')
model.summary() 
img_path ="Downloads/DSCN27770.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
plt.imshow(img_tensor[0])
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

layer_names
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]
    
    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]
    
    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    
    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


plt.show()
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
#import collections
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

import matplotlib.pyplot as plt
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

import numpy as np
np.random.seed(100)
img_width, img_height = 150, 150
epochs = 10
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:\Users\nnidamanuru\Downloads\GE_ML',
                                                                                             target_base_dir='C:\Users\nnidamanuru\Downloads\GE_ML_neh')
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:\Users\nnidamanuru\Downloads\GE_ML',\
                                                                                             target_base_dir='C:\Users\nnidamanuru\Downloads\GE_ML_neh')
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)
save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil

def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 10
batch_size = 20

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)
save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
from keras import applications,callbacks
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg162'
preprocess = applications.vgg16.preprocess_input
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil




def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 10
batch_size = 20

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')

print(train_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)
save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
epochs = 200
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)

def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg161'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)

## ---(Thu May 17 15:11:13 2018)---
import os
os.getcwd()
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg161'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (58) + [1] * (57))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (14) + [1] * (14))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.add(Dense(2, activation='softmax'))
print(model.summary())
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)
save_weights = ModelCheckpoint('bottlenck_model_vgg16.h5', monitor='val_loss', save_best_only=True)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
X_train.shape[1:]
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg16/bottlenck_model_vgg16.h5'
train_dir, validation_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))
top_model.load_weights(top_model_weights_path)
model = Model(inputs = base_model.input, outputs = top_model(base_model.output))
set_trainable = False
for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.24,
        shear_range=0.21,
        zoom_range=0.27,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
##e
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('model11705vgg16.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
 Importing modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = model.predict_generator(test_generator, 26//5)
probabilities
mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1

file
test_generator.filenames
file.split('\\')[1]
file.split('\\')[1].split('.')[0]
file.split('\\')[1].split('.')[1]
id 
mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1

mapper
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_neh')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
test_dir='C:\\Users\\nnidamanuru\\Downloads\\ML_test'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#prin
test_dir='C:\\Users\\nnidamanuru\\Downloads\\ML_test'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = model.predict_generator(test_generator, 26//5)
probabilities
mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i]
    i += 1

mapper
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_neh_vgg')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_vgg16'
names1=['Drum by the side of crate', 'Drum by the side of multiple crate', 'Drum on single crate', 'Drum on single crate along with noise', 'Only Drum', 'Only single crate']
print(names1.sort())
names1=['Drum by the side of crate', 'Drum by the side of multiple crate', 'Drum on single crate', 'Drum on single crate along with noise', 'Only Drum', 'Only single crate']
names1.sort()
names1
words = [ 'baloney', 'aardvark' ]
words.sort()
words
import cv2
import cv2
from keras.preprocessing import image
x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro'
img = image.load_img(img_path, target_size=(224, 224))
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img
img = image.img_to_array(img)
img
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
import cv2
x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
img
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
crop_img
crop_img = img[y1:y2, x1:x2]

## ---(Fri May 18 12:56:06 2018)---
import cv2
x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
crop_img
cv2.imshow('img',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## ---(Fri May 18 12:56:06 2018)---
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
img_path
cv2.destroyAllWindows()
train_dir_original='C:/Users/nnidamanuru/Downloads/ML'
target_base_dir='C:/Users/nnidamanuru/Downloads/ML_cropped'
train_dir = os.path.join(target_base_dir, 'train')
import os
import shutil
import random
train_dir = os.path.join(target_base_dir, 'train')
train_dir
validation_dir = os.path.join(target_base_dir, 'validation')
validation_dir
categories = os.listdir(train_dir_original)
categories
print(categories)
import scipy.misc
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    x1 = 54
    y1 = 296
    x2 = 932
    y2 = 626
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                img = cv2.imread(t)
                crop_img = img[y1:y2, x1:x2]
                img=scipy.misc.toimage(crop_img)
                shutil.copy2(img, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                img = cv2.imread(v)
                crop_img = img[y1:y2, x1:x2]
                img=scipy.misc.toimage(crop_img)
                shutil.copy2(img, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_cropped')
train_dir_original='C:/Users/nnidamanuru/Downloads/ML'
target_base_dir='C:/Users/nnidamanuru/Downloads/ML_cropped'
train_dir = os.path.join(target_base_dir, 'train')
validation_dir = os.path.join(target_base_dir, 'validation')
categories = os.listdir(train_dir_original)
print(categories)
x1 = 54
y1 = 296
x2 = 932
y2 = 626
if not os.path.exists(target_base_dir):          
    os.mkdir(target_base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    
    for c in categories:
        train_dir_original_cat = os.path.join(train_dir_original, c)
        files = os.listdir(train_dir_original_cat)
        train_files = [os.path.join(train_dir_original_cat, f) for f in files]
        random.shuffle(train_files)    
        n = int(len(train_files) * val_percent)
        val = train_files[:n]
        train = train_files[n:]  
        
        train_category_path = os.path.join(train_dir, c)
        os.mkdir(train_category_path)
        for t in train:
            img = cv2.imread(t)
            crop_img = img[y1:y2, x1:x2]
            img=scipy.misc.toimage(crop_img)
            shutil.copy2(img, train_category_path)
        
        val_category_path = os.path.join(validation_dir, c)
        os.mkdir(val_category_path)
        for v in val:
            img = cv2.imread(v)
            crop_img = img[y1:y2, x1:x2]
            img=scipy.misc.toimage(crop_img)
            shutil.copy2(img, val_category_path)

val_percent=0.2
if not os.path.exists(target_base_dir):          
    os.mkdir(target_base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    
    for c in categories:
        train_dir_original_cat = os.path.join(train_dir_original, c)
        files = os.listdir(train_dir_original_cat)
        train_files = [os.path.join(train_dir_original_cat, f) for f in files]
        random.shuffle(train_files)    
        n = int(len(train_files) * val_percent)
        val = train_files[:n]
        train = train_files[n:]  
        
        train_category_path = os.path.join(train_dir, c)
        os.mkdir(train_category_path)
        for t in train:
            img = cv2.imread(t)
            crop_img = img[y1:y2, x1:x2]
            img=scipy.misc.toimage(crop_img)
            shutil.copy2(img, train_category_path)
        
        val_category_path = os.path.join(validation_dir, c)
        os.mkdir(val_category_path)
        for v in val:
            img = cv2.imread(v)
            crop_img = img[y1:y2, x1:x2]
            img=scipy.misc.toimage(crop_img)
            shutil.copy2(img, val_category_path)

img
img=scipy.misc.toimage(crop_img)
import PIL
import cv2
import os
import shutil
import random
import scipy.misc
import PIL

x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
img=cv2.imwrite(crop_img)
img=cv2.imwrite('crop_img',crop_img)
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    x1 = 54
    y1 = 296
    x2 = 932
    y2 = 626
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                img = cv2.imread(t)
                crop_img = img[y1:y2, x1:x2]
                
                shutil.copy2(crop_img, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                img = cv2.imread(v)
                crop_img = img[y1:y2, x1:x2]
                
                shutil.copy2(crop_img, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_cropped')

import cv2
import os
import shutil
import random
import scipy.misc
import PIL

x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
img=cv2.imwrite('C:/Users/nnidamanuru/Downloads/New folder',crop_img)
img=cv2.imwrite('C:/Users/nnidamanuru/Downloads/New folder/mypic.jpg',crop_img)
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    x1 = 54
    y1 = 296
    x2 = 932
    y2 = 626
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                img = cv2.imread(t)
                img = img[y1:y2, x1:x2]
                
                cv2.imwrite(os.path.join(train_category_path, t),img)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                img = cv2.imread(v)
                img = img[y1:y2, x1:x2]
                
                cv2.imwrite(os.path.join(val_category_path, v),img)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_cropped')
t
os.path.join(train_category_path, t)
train_category_path
path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate'
os.listdir(path)
path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate'
dest='C:/Users/nnidamanuru/Downloads/New folder'
path1 = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate'
dest='C:/Users/nnidamanuru/Downloads/New folder'
for i in os.listdir(path1):
    img = cv2.imread(i)
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(dest, i),crop_img)

import cv2
import os
import shutil
import random
import scipy.misc
import PIL

x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate'
dest='C:/Users/nnidamanuru/Downloads/New folder'

for i in os.listdir(path1):
    img = cv2.imread(i)
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(dest, i),crop_img)

for i in os.listdir(path1):
    img = cv2.imread(os.path.join(dest, i))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(dest, i),crop_img)

for i in os.listdir(path1):
    img = cv2.imread(os.path.join(path1, i))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(dest, i),crop_img)

cv2.imshow('img',crop_img)
os.listdir(path1)

## ---(Fri May 18 14:31:17 2018)---
import cv2
import os
import shutil
import random
import scipy.misc
import PIL

x1 = 54
y1 = 296
x2 = 932
y2 = 626

path1 = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate'
dest='C:/Users/nnidamanuru/Downloads/New folder'
os.listdir(path1)
os.path.join(path1, i)
for i in os.listdir(path1):
    img = cv2.imread(os.path.join(path1, i))

os.path.join(path1, i)
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate'
dest='C:\\Users\\nnidamanuru\\Downloads\\New folder'
for i in os.listdir(path1):
    img = cv2.imread(os.path.join(path1, i))
    img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(dest, i),img)

import cv2
import os
import shutil
import random
import scipy.misc
import PIL

x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
dest='C:\\Users\\nnidamanuru\\Downloads\\New folder'
img=cv2.imread(path1)
img=img[y1:y2, x1:x2]
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro.jpg',img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)

## ---(Fri May 18 14:43:28 2018)---
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)

## ---(Fri May 18 15:04:12 2018)---
import cv2
x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
import cv2
x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
img
crop_img = img[y1:y2, x1:x2]
crop_img
cv2.imshow("cropped", crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro.jpg',crop_img)
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro1.jpg',crop_img)
import cv2
x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro1.jpg',crop_img)
import cv2
from PIL import Image
x1 = 54
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
.
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im
pil_im.show()
x1 = 540
y1 = 296
x2 = 932
y2 = 626
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
pil_im
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\ML\\Drum by the side of crate\\WIN_20180516_17_01_16_Pro.jpg'
img=cv2.imread(path1)
cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
pil_im
pil_im.show()
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\GE_ML\\Negative\\WIN_20180516_16_52_00_Pro.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\GE_ML\\Negative\\WIN_20180516_16_52_00_Pro.jpg'
img=cv2.imread(path1)
y
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro2.jpg',crop_img)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_neh_CROPPED')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)
save_weights = ModelCheckpoint('model11705.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples

def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/ML_neh_CROPPED_vgg16')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (19) + [1] * (9) +[2]*(12)+[3]*36 +[4]*8+[5]*14)
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (4) + [1] * (2) +[2]*(3)+[3]*9 +[4]*2+[5]*3)
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(6, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
sa
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001) 
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil




def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped')
print(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705_cropped.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
words = [ 'Positive', 'Negative','Other' ]
words.sort()
words
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (36) + [1] * (14) + [2] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (8) + [1] * (3) + [2] * (12))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16lay.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(3, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16lay.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN')
print(train_dir)
model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (49) + [1] * (11) + [2] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (12) + [1] * (1) + [2] * (12))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (12) + [1] * (2) + [2] * (12))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other_vgg16/bottlenck_model_cropped_vgg16other.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
top_model.load_weights(top_model_weights_path)
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN')

print(train_dir)

model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (49) + [1] * (11) + [2] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (12) + [1] * (2) + [2] * (12))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other1.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other2.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()



img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other_vgg16/bottlenck_model_cropped_vgg16other2.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
top_model.load_weights(top_model_weights_path)
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16/bottlenck_model_cropped_vgg16other2.h5'
ase_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
top_model.load_weights(top_model_weights_path)
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN')

print(train_dir)

model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))

bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16'
preprocess = applications.vgg16.preprocess_input

save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)

X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (49) + [1] * (11) + [2] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape

X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (12) + [1] * (2) + [2] * (12))
y_validation = np_utils.to_categorical(validation_labels)

X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(3, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16other3.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()



img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16/bottlenck_model_cropped_vgg16other3.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
top_model.load_weights(top_model_weights_path)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
top_model.load_weights(top_model_weights_path)
model = Model(inputs = base_model.input, outputs = top_model(base_model.output))
set_trainable = False
for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.24,
        shear_range=0.21,
        zoom_range=0.27,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('model11705vgg16fine.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil

## ---(Mon May 21 12:34:58 2018)---
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
model=Sequential()
model.load_weights('Nehru Nidamanuru/model11705.h5')
model.load_weights('Nehru Nidamanuru\\model11705.h5')
model.load_weights('Downloads\\model11705.h5')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16/bottlenck_model_cropped_vgg16other3.h5'

train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))

top_model.load_weights(top_model_weights_path)
top_model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = top_model.predict_generator(test_generator, 26//5)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras import backend as K
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras import backend as K
import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16/bottlenck_model_cropped_vgg16other3.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

top_model = Sequential()
top_model.add(Flatten(input_shape=input_shape))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))
top_model.load_weights(top_model_weights_path)
print(top_model.summary())
top_model.load_weights(top_model_weights_path)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras import backend as K
import random
import shutil


#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='Downloads\\model11705.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

top_model = Sequential()
top_model.add(Flatten(input_shape=input_shape))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))
top_model.load_weights(top_model_weights_path)
top_model = Sequential()
top_model.add(Flatten(input_shape=input_shape))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))
top_model.load_weights(top_model_weights_path)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras import backend as K
import random
import shutil
# Importing modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
#import collections
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()



img_width, img_height = 150, 150
epochs = 10
batch_size = 20
top_model_weights_path ='Downloads\\model11705.h5'
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML_Cropped',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_neh_Cropped_vgg16_otherN_fine')
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
print(model.summary())

model.load_weights(top_model_weights_path)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout, Flatten, Dense
from keras.layers import Flatten, Dense
from keras import backend as K
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from keras import applications,callbacks
from keras.utils import np_utils



import os
import random
import shutil
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 10
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='/var/www/projects/nehru/monilia/old/GE_ML_2class')
pr
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_2class')
print(train_dir)
os.listdir(train_dir)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(input_shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705_2class.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
print(model.summary())
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.27,
        height_shift_range=0.27,
        shear_range=0.24,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705_2class.h5', monitor='val_loss', save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())


model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights('Downloads\\model11705_2class.h5')
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = model.predict_generator(test_generator, 26//5)
probabilities
len(probabilities)
test_generator
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())


model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights('Downloads\\model11705_2class.h5')
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2, activation='softmax'))
print(model1.summary())


model1.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model1.load_weights('Downloads\\model11705_2class.h5')
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = model1.predict_generator(test_generator, 26//5)
len(probabilities)
mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1

mapper.values()
mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i]
    i += 1

tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission.csv', columns=['id','label'], index=False)
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2, activation='softmax'))
print(model1.summary())


model1.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model1.load_weights('Downloads\\model11705_2clas.h5')
test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
probabilities = model1.predict_generator(test_generator, 26//5)
len(probabilities)
mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i]
    i += 1

tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission1.csv', columns=['id','label'], index=False)
from keras.models import model_from_json
model1 = model_from_json('Downloads\\model')
model1 = model_from_json('Downloads\\model.json')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
history = model.fit_generator(
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
import random
import shutil



#Data preparing for Train and Validation
def preapare_dataset_for_flow(train_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                shutil.copy2(t, train_category_path)
            
            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                shutil.copy2(v, val_category_path)
    
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    return train_dir, validation_dir, nb_train_samples, nb_validation_samples


#To plot the loss and accuracy of Train and Validation
def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


##Save the weights from pretrained VGG16  
def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, ((nb_train_samples // batch_size)+1))
        np.save(open('bottleneck_features_train_vgg16.npy', 'wb'),
               bottleneck_features_train)
        
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, ((nb_validation_samples // batch_size)+1))
        np.save(open('bottleneck_features_validation_vgg16.npy', 'wb'),
                bottleneck_features_validation)
    
    
    else:
        print('bottleneck directory already exists')


np.random.seed(100)
img_width, img_height = 150, 150
epochs = 200
batch_size = 20
train_dir, valid_dir, nb_train_samples, nb_validation_samples=preapare_dataset_for_flow(train_dir_original='C:/Users/nnidamanuru/Downloads/GE_ML',
                                                                                             target_base_dir='C:/Users/nnidamanuru/Downloads/GE_ML_2class')
print(train_dir)

model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/bottleneck_features_cropped_ge_Other1_vgg16_2class'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, valid_dir,bottleneck_dir)
X_train = np.load(open('bottleneck_features_train_vgg16.npy','rb'))
train_labels = np.array([0] * (49) + [1] * (11) + [2] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
train_labels = np.array([0] * (47) + [1] * (48))
y_train = np_utils.to_categorical(train_labels)
X_train.shape
y_train.shape
X_validation = np.load(open('bottleneck_features_validation_vgg16.npy','rb'))
validation_labels = np.array([0] * (11) + [1] * (11))
y_validation = np_utils.to_categorical(validation_labels)
X_validation.shape
y_validation.shape
p=0.5
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)   
save_weights = ModelCheckpoint('bottlenck_model_cropped_vgg16_2class.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
              epochs=200,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, reduce_lr])
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json) 

from keras.models import model_from_json
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
import os
import cv2
import math


def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
        return   
    os.makedirs(target_dir)      
    categories = os.listdir(base_dir)
    for category in categories:
        video_category_path = os.path.join(base_dir, category)    
        video_listings = os.listdir(video_category_path)
        frames_category_path = os.path.join(target_dir, category)
        count = 1
        for file in video_listings[0:2]:
            video = cv2.VideoCapture(os.path.join(video_category_path,file))
            #print(video.isOpened())
            framerate = video.get(5)
            frames_path = os.path.join(frames_category_path, "video_" + str(int(count)))
            os.makedirs(frames_path)
            while (video.isOpened()):
                frameId = video.get(1)
                success,image = video.read()
                if(success == False):
                    break
                image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
                if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(frames_path, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                    print(filename)
                    cv2.imwrite(filename,image)
            video.release()
            count+=1

preprocess_videos('C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos', 
                  'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data2')
preprocess_videos('C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos', 
                  'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3')

## ---(Mon May 21 18:34:29 2018)---
print(keras.__version__)
print(tensorflow.version)
print(tensorflow.__version__)
import tensorflow as tf
from ssd import SSD300
import os
os.chdir('Nehru Nidamanuru')
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np
Labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
               'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard'
               , 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
len(Labels)
NUM_CLASSES = len(Labels) + 1
input_shape=(300, 300, 3)
weights_dir = 'C:/Users/nnidamanuru/Downloads/VGG_coco_SSD_300x300_iter_400000.h5'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)
input_shape
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)
weights_dir = 'C:/Users/nnidamanuru/Downloads/weights_SSD300.hdf5'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)
path = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1'
batch_size = 1
target_size = (300, 300)
nb_test_samples = 5
test_generator = generator.DirectoryIterator(directory=path,  class_mode=None,
                target_size=target_size, batch_size=batch_size,  shuffle=False)
test_generator
preds = model.predict_generator(test_generator, nb_test_samples//batch_size)
print (np.shape(preds))
preds_1 = preds[0,0,:]
print (preds_1)
nms_thresh = 0.4
bbox_util = ssd_utils.BBoxUtility(NUM_CLASSES,nms_thresh = nms_thresh)
results = bbox_util.detection_out(preds)
print (np.shape(results))
print (results[0].shape)
print (results[0][0])
images = next(test_generator)
results[4]
results
enumerate(next(test_generator))
images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)
images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)
score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh)

for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)

## ---(Tue May 22 13:22:43 2018)---
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)

weights_dir = 'C:/Users/nnidamanuru/Downloads/weights_SSD300.hdf5'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)

path = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1'
batch_size = 1
target_size = (300, 300)
nb_test_samples = 5
test_generator = generator.DirectoryIterator(directory=path,  class_mode=None,
                target_size=target_size, batch_size=batch_size,  shuffle=False)

preds = model.predict_generator(test_generator, nb_test_samples//batch_size)
print (np.shape(preds))
preds_1 = preds[0,0,:]
print (preds_1)
preds_1 = preds[0,:]
print (preds_1)
preds_1 = preds[:]
print (preds_1)
preds_1 = preds[0,0,:]
print (preds_1)
nms_thresh = 0.4
bbox_util = ssd_utils.BBoxUtility(NUM_CLASSES,nms_thresh = nms_thresh)
results = bbox_util.detection_out(preds_1)
results = bbox_util.detection_out(preds)
results
print (np.shape(results))
print (results[0].shape)
print (results[0][0])
print (results[0][1])
print (results[0,0,:][1])
print (results[2:][1])
print (results[2][1])
print (results[0][0])
images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)
images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)
score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)

images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np

voc_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)

weights_dir = 'C:/Users/nnidamanuru/Downloads/VGG_coco_SSD_300x300_iter_400000.h5'
NUM_CLASSES
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)
weights_dir = 'C:/Users/nnidamanuru/Downloads/VGG_coco_SSD_300x300_iter_400000.h5'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np

voc_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)

weights_dir = 'C:/Users/nnidamanuru/Downloads/VGG_coco_SSD_300x300_iter_400000.h5'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)
import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np

voc_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)

weights_dir = 'C:/Users/nnidamanuru/Downloads/yolo.weights'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)

## ---(Wed May 23 11:33:38 2018)---
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
os.chdoir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')
os.chdir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_60.jpg'
input_image = Image.open("images/" + input_image_name)
input_image = Image.open("images\\" + input_image_name)
input_image = Image.open(input_image_name)
width, height = input_image.size
width, height
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
image_shape = (height, width)
image_shape
class_names = read_classes("model_data/coco_classes.txt")
class_names
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_model.output
anchors
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
yolo_outputs
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
boxes, scores, classes
boxes
sess = K.get_session()
image, image_data = preprocess_image("images/" + input_image_name, model_image_size = (608, 608))
image, image_data = preprocess_image(input_image_name, model_image_size = (608, 608))
image, image_data
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
yolo_model.input
image_data
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
yolo_model.summary()
image_shape
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
image.save(os.path.join("out", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out", input_image_name))
imshow(output_image)
os.path
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
import os

from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

os.chdir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_220.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
image_shape = (height, width)
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
sess = K.get_session()
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_63.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_path = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1'
os.listdir(input_image_path)
import os

from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

os.chdir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval


#Provide the name of the image that you saved in the images folder to be fed through the network
input_image_path = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1'
for i in os.listdir(input_image_path):
    input_image_name=os.path.join(i,input_image_path)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name)
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

for i in os.listdir(input_image_path):
    input_image_name=os.path.join(i,input_image_path)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name,'wb')
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

for i in os.listdir(input_image_path):
    input_image_name=os.path.join(i,input_image_path)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name,'rb')
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

input_image_name
for i in os.listdir(input_image_path):
    input_image_name=os.path.join(i,input_image_path)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name,"rb")
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

for i in os.listdir(input_image_path):
    input_image_name=os.path.join(i,input_image_path)

input_image_name
i
for i in os.listdir(input_image_path):
    input_image_name=os.path.join(input_image_path,i)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name,"rb")
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

for i in os.listdir(input_image_path):
    input_image_name=os.path.join(input_image_path,i)
    
    
    #Obtaining the dimensions of the input image
    input_image = Image.open(input_image_name)
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    
    #Assign the shape of the input image to image_shapr variable
    image_shape = (height, width)
    
    
    #Loading the classes and the anchor boxes that are provided in the madel_data folder
    class_names = read_classes("model_data/coco_classes.txt")
    len(class_names)
    anchors = read_anchors("model_data/yolo_anchors.txt")
    
    #Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
    yolo_model = load_model("model_data/yolo.h5")
    
    #Print the summery of the model
    yolo_model.summary()
    
    #Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    
    
    # Initiate a session
    sess = K.get_session()
    
    
    #Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
    
    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    
    
    #Print the results
    print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
    image.save(os.path.join("out1", input_image_name), quality=90)
    output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
    imshow(output_image)

input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_1'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_1.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_60.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_59.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master\\images'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master\\images\\test4.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name,)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_61.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1\\video_1\\image_62.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
from PIL import Image
import cv2

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_1.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
cv2.imshow("cropped", crop_img)
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img=cv2.imread(path1)

# Croping
crop_img = img[y1:y2, x1:x2]
cv2.imshow("cropped", crop_img)
cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
import cv2
x1 = 732
y1 = 168
x2 = 1405
y2 = 1053
img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626
img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
pil_im.show()
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\WIN_20180516_17_01_16_Pro2.jpg',crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\image_88.jpg',crop_img)
import cv2
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 732
y1 = 168
x2 = 1405
y2 = 1053


img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077



img_path = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img = cv2.imread(img_path)
#img = image.img_to_array(img)




# Croping
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
import cv2

x1 = 54
y1 = 296
x2 = 932
y2 = 626

img_path = 'C:/Users/nnidamanuru/Downloads/ML/Drum by the side of crate/WIN_20180516_17_01_16_Pro.jpg'
img = cv2.imread(img_path)
#img = image.img_to_array(img)




# Croping
crop_img = img[y1:y2, x1:x2]
cv2.imshow('img',crop_img)
from PIL import Image
import cv2
x1 = 484
y1 = 314
x2 = 1063
y2 = 1077
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_88.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
import os
import cv2
import math


def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
        return   
    os.makedirs(target_dir)      
    categories = os.listdir(base_dir)
    for category in categories:
        video_category_path = os.path.join(base_dir, category)    
        video_listings = os.listdir(video_category_path)
        frames_category_path = os.path.join(target_dir, category)
        count = 1
        for file in video_listings[0:2]:
            video = cv2.VideoCapture(os.path.join(video_category_path,file))
            #print(video.isOpened())
            framerate = video.get(5)
            frames_path = os.path.join(frames_category_path, "video_" + str(int(count)))
            os.makedirs(frames_path)
            while (video.isOpened()):
                frameId = video.get(1)
                success,image = video.read()
                if(success == False):
                    break
                
                if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(frames_path, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                    print(filename)
                    cv2.imwrite(filename,image)
            video.release()
            count+=1


preprocess_videos('C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos', 
                  'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4')
from PIL import Image
import cv2

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077


path1 = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_62.jpg'
img=cv2.imread(path1)
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()
cv2.imshow("cropped", crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\image_62.jpg',crop_img)
import shutil
path2='C:\\Users\\nnidamanuru\\Downloads\\neh\\test'
os.listdir(path2)
for i in os.listdir(path2):
    path3=os.path.join(path2,i)

path3
path2='C:\\Users\\nnidamanuru\\Downloads\\neh\\test'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path1)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path1)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


## ---(Wed May 23 16:21:07 2018)---
from PIL import Image
import cv2
import shutil
import os

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077
path2='C:\\Users\\nnidamanuru\\Downloads\\neh\\test'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path1)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

from PIL import Image
import cv2
import shutil
import os
in_path='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4'
os.listdir(in_path)
os.listdir(in_path1)
in_path='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1'
os.listdir(in_path)
in_path='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4'
for i in os.listdir(in_path):
    in_path1=os.path.join(in_path,i)
    for i in os.listdir(in_path1):
        in_path1=os.path.join(in_path1,i)

os.listdir(in_path1)
in_path1
from PIL import Image
import cv2
import os
x1 = 732
y1 = 168
x2 = 1405
y2 = 1053
path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario1\\1'
os.listdir(path2)
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\2\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario1\\2'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario1\\3'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 657
y1 = 109
x2 = 1431
y2 = 1078
path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario2\\1'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\2\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario2\\2'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario2\\3'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077
path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario3\\1'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\2\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario3\\2'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario3\\3'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
x1 = 445
y1 = 253
x2 = 1063
y2 = 1077
path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario4\\1'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\2\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario4\\2'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario4\\3'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


## ---(Thu May 24 11:32:01 2018)---
import os

from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

os.chdir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_49.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
image_shape = (height, width)
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
sess = K.get_session()
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
class_names
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\refrigerator_PNG9060.PNG'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
width, height
height = np.array(height, dtype=float)
image_shape = (height, width)
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
image_data
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\LG_refrigerator.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
image_shape = (height, width)
image_shape 
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")
#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
sess = K.get_session()
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\refrigerator-box.jpg'
#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)
#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")
#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
# Initiate a session
sess = K.get_session()
#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_60.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_61.jpg'
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_62.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
import os

from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

os.chdir('C:\\Users\\nnidamanuru\\Downloads\\YOLOw-Keras-master')

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval

##\\Ge_videos\\data4\\1\\video_1\\image_49.jpg
#Provide the name of the image that you saved in the images folder to be fed through the network
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_62.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_63.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})

print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out1", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out1", input_image_name))
imshow(output_image)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_64.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_65.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
input_image_name = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\1\\video_1\\image_66.jpg'

#Obtaining the dimensions of the input image
input_image = Image.open(input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
len(class_names)
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image(input_image_name, model_image_size = (416, 416))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
import os
import math
import cv2



def get_store_frames(video_file, video_frames_dir):
     print(video_file, video_frames_dir)
     video = cv2.VideoCapture(video_file)
     #print(video.isOpened())
     framerate = video.get(5)
     os.makedirs(video_frames_dir)
     while (video.isOpened()):
         frameId = video.get(1)
         success,image = video.read()
         if(success == False):
             break
         if (frameId % math.floor(framerate) == 0):
                filename = os.path.join(video_frames_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                print(filename)
                cv2.imwrite(filename,image)
     video.release()


def preapare_full_dataset_for_flow(train_dir_original,target_base_dir):
    train_dir = os.path.join(target_base_dir, 'train')
    
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train_files:
                frames_dir = t.split('\\')[-1].split('.')[0]
                get_store_frames(t, os.path.join(train_category_path, frames_dir))
    
    
    else:
        print('required directory structure already exists. learning continues with existing data')
    
    nb_train_samples = 0  
    
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training samples:', nb_train_samples)
    
    return train_dir,nb_train_samples


train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datan')
train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data1')
video_frames_dir
train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA
train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA'
target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data2'
train_dir = os.path.join(target_base_dir, 'train')
train_dir
categories = os.listdir(train_dir_original)
print(categories)
import os
import math
import cv2



def get_store_frames(video_file, video_frames_dir):
     print(video_file, video_frames_dir)
     video = cv2.VideoCapture(video_file)
     #print(video.isOpened())
     framerate = video.get(cv2.CAP_PROP_FPS)
     os.makedirs(video_frames_dir)
     while (video.isOpened()):
         frameId = video.get(1)
         success,image = video.read()
         if(success == False):
             break
         if (frameId % math.floor(framerate) == 0):
                filename = os.path.join(video_frames_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                print(filename)
                cv2.imwrite(filename,image)
     video.release()


def preapare_full_dataset_for_flow(train_dir_original,target_base_dir):
    train_dir = os.path.join(target_base_dir, 'train')
    
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            
            
            for t in train_files:
                frames_dir = t.split('\\')[-1].split('.')[0]
                get_store_frames(t, os.path.join(target_base_dir, frames_dir))
    
    
    else:
        print('required directory structure already exists. learning continues with existing data')
    
    nb_train_samples = 0  
    
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training samples:', nb_train_samples)
    
    return train_dir,nb_train_samples

train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data2\\images')
train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data2')
n=images
import os
import math
import cv2


n='images'
def get_store_frames(video_file, video_frames_dir):
     print(video_file, video_frames_dir)
     video = cv2.VideoCapture(video_file)
     #print(video.isOpened())
     framerate = video.get(cv2.CAP_PROP_FPS)
     os.makedirs(video_frames_dir)
     while (video.isOpened()):
         frameId = video.get(1)
         success,image = video.read()
         if(success == False):
             break
         if (frameId % math.floor(framerate) == 0):
                filename = os.path.join(video_frames_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                print(filename)
                cv2.imwrite(filename,image)
     video.release()


def preapare_full_dataset_for_flow(train_dir_original,target_base_dir):
    train_dir = os.path.join(target_base_dir, 'train')
    
    categories = os.listdir(train_dir_original)
    print(categories)
    
    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        
        
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            
            
            for t in train_files:
                get_store_frames(t, os.path.join(target_base_dir, n))
    
    
    else:
        print('required directory structure already exists. learning continues with existing data')
    
    nb_train_samples = 0  
    
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training samples:', nb_train_samples)
    
    return train_dir,nb_train_samples

train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data3')
preprocess_videos(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos', 
base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos'
target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4'
import os
import cv2
import math
if not os.path.exists(base_dir): 
    print(base_dir, ' does not exist')
    return

target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datax'
os.makedirs(target_dir)      
categories = os.listdir(base_dir)
categories
base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1
base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1'
categories = os.listdir(base_dir)
categories
import os
import math
import cv2
import random

for i in range(50):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
    success,image = vidcap.read()
    cv2.imwrite("frames2/frame%d.jpg" % count, image)     # save frame as JPEG file
    print ('Read a new frame: ', success)
    count += 1

def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    for file in video_listings[0:1]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        success,image = vidcap.read()
        count = 0
        success = True
        for i in range(5):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    
            success,image = vidcap.read()
            cv2.imwrite(target_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file
            print ('Read a new frame: ', success)
            count += 1
    return

import os
import math
import cv2
import random

for i in range(50):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
    success,image = vidcap.read()
    cv2.imwrite("frames2/frame%d.jpg" % count, image)     # save frame as JPEG file
    print ('Read a new frame: ', success)
    count += 1

import os
import math
import cv2
import random

def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    for file in video_listings[0:1]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        success,image = vidcap.read()
        count = 0
        success = True
        for i in range(5):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    
            success,image = vidcap.read()
            cv2.imwrite(target_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file
            print ('Read a new frame: ', success)
            count += 1
    return

preprocess_videos(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\dataj')
base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1'
target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datak'
video_listings = os.listdir(base_dir)
print(video_listings)
import os
import math
import cv2
import random

def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    for file in video_listings[0:2]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        success,image = vidcap.read()
        count = 0
        success = True
        for i in range(5):
            vidcap.set(cv2.CAP_PROP_FPS)    
            success,image = vidcap.read()
            cv2.imwrite(target_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file
            print ('Read a new frame: ', success)
            count += 1
    return

video_listings[0:1]
video_listings[0:2]
import os
import math
import cv2
import random

def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    for file in video_listings[0:3]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        success,image = vidcap.read()
        count = 0
        success = True
        for i in range(5):
            vidcap.set(cv2.CAP_PROP_FPS)    
            success,image = vidcap.read()
            cv2.imwrite(target_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file
            print ('Read a new frame: ', success)
            count += 1
    return

preprocess_videos(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datak')
import os
import math
import cv2
def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    for file in video_listings[0:3]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
        while (vidcap.isOpened()):
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if(success == False):
                break
            if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(target_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                    print(filename)
                    cv2.imwrite(filename,image)
        vidcap.release()
    return

preprocess_videos(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datak')
math.floor(framerate))
def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    count=0
    for file in video_listings[0:3]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
        while (vidcap.isOpened()):
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if(success == False):
                break
            if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(target_dir, "image_" + str(count) + ".jpg")
                    count+=1
                    print(filename)
                    cv2.imwrite(filename,image)
        vidcap.release()
    return

preprocess_videos(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Videos1\\1', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\datap')
561/60
import os
import math
import cv2


def get_store_frames(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    count=0
    for file in video_listings[0:len(video_listings)]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
        while (vidcap.isOpened()):
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if(success == False):
                break
            if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(target_dir, "image_" + str(count) + ".jpg")
                    count+=1
                    print(filename)
                    cv2.imwrite(filename,image)
        vidcap.release()
    return


get_store_frames(base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Videos', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Images_fps')

## ---(Fri May 25 21:30:05 2018)---
import os
import cv2
import math


def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
        return   
    os.makedirs(target_dir)      
    categories = os.listdir(base_dir)
    for category in categories:
        video_category_path = os.path.join(base_dir, category)    
        video_listings = os.listdir(video_category_path)
        frames_category_path = os.path.join(target_dir, category)
        count = 1
        for file in video_listings[0:2]:
            video = cv2.VideoCapture(os.path.join(video_category_path,file))
            #print(video.isOpened())
            framerate = video.get(cv2.CAP_PROP_FPS)
            frames_path = os.path.join(frames_category_path, "video_" + str(int(count)))
            os.makedirs(frames_path)
            while (video.isOpened()):
                frameId = video.get(1)
                success,image = video.read()
                if(success == False):
                    break
                
                if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(frames_path, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                    print(filename)
                    cv2.imwrite(filename,image)
            video.release()
            count+=1

preprocess_videos('C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\Video', 
                  'C:\\Users\\nnidamanuru\\Downloads\\GEA\\test')
preprocess_videos('C:\\Users\\nnidamanuru\\Downloads\\GEA_3_class\\video', 
                  'C:\\Users\\nnidamanuru\\Downloads\\GEA_3_class\\non_box')

## ---(Mon May 28 12:13:17 2018)---
import os
import math
import cv2


def get_store_frames(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    count=0
    for file in video_listings[0:len(video_listings)]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
        while (vidcap.isOpened()):
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if(success == False):
                break
            if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(target_dir, "image_" + str(count) + ".jpg")
                    count+=1
                    print(filename)
                    cv2.imwrite(filename,image)
        vidcap.release()
    return

get_store_frames(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Bander', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA_3\\Images_fps')

## ---(Thu May 31 16:14:12 2018)---
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
img_width, img_height=300,300
batch_size=20
model = load_model('model_2class1.json')
from keras.models import model_from_json
model = model_from_json('Downloads\\model_2class1.json')
from keras.models import model_from_json
with open('Downloads\\model_2class1.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('Downloads\\model23105_2class.h5')
print(model.summary())
test_dir='C:\\Users\\nnidamanuru\\Pictures\\Saved Pictures'
test_datagen = ImageDataGenerator(rescale=1. / 255)
img_width, img_height=150,150
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
test_dir='C:\\Users\\nnidamanuru\\Downloads\\drum\\test1'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#prin
probabilities = model.predict_generator(test_generator, 16//16)
print(probabilities)
mapper = {}
i = 0
for file in test_generator.filenames:
    id = str(file.split('/')[1])
    mapper[id] = probabilities[i]
    i += 1

mapper = {}
i = 0
for file in test_generator.filenames:
    id = str(file.split('\\')[1])
    mapper[id] = probabilities[i]
    i += 1

tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission_neh.csv', columns=['id','label'], index=False)
from keras.models import load_model
from keras.models import model_from_json
from keras.models import load_model
from keras.models import model_from_json
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications,callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras.preprocessing import  image
from keras.utils import np_utils
import random
import shutil
import matplotlib.pyplot as plt
img_width, img_height=300,300
batch_size=20
# Model reconstruction from JSON file
with open('Downloads\\model_3class1svm.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('Downloads\\bottlenck_model_vgg_3_class_svm.h5')
print(model.summary())
test_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\test\\1'
def save_bottlebeck_features1(model, preprocess_fn,test_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)
        
        test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_test = model.predict_generator(
                test_generator, (176 // 14) + 1)
        np.save(open('bottleneck_features_test_vgg_sklearn_train.npy', 'wb'), bottleneck_features_test)
    else:
        print('bottleneck directory already exists')

model_1 = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
b
model_1 = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
bottleneck_dir = 'C:/Users/nnidamanuru/Downloads/GEA/bottleneck_features_vgg_3_one_class_svm1_final_neh13'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features1(model_1, preprocess,test_dir,bottleneck_dir)

## ---(Sat Jun  2 16:06:55 2018)---
from keras import Input,Dense
from keras.layers import Input,Dense
from keras.datasets import mnist
from keras.models import Model
(X_train,_),(X_test,_)=mnist.load_data()
X_train=X_train.astype('float32') / 255.
X_test=X_test.astype('float32') / 255.
from keras.models import Model
print(X_test.shape)
X_test=X_test.reshape(X_test.shape[0], -1)
X_test.shape
X_test=X_test.reshape((X_test.shape[0], -1))
X_test.shape
X_train=X_train.reshape(X_train.shape[0],-1)
X_train.shape
input_size=784
encoding_size=64
batchsize=200
epochs=10
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='categorical_crossentropy')
autoencoder.fit(X_train,X_train,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(X_train,X_train,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
deosde_imgs=autoencoder.predict(X_test)
encoder=Model(input_image,decoded)
tmp=encoder.predict(X_test)
encoder=Model(input_image,encoded)
tmp=encoder.predict(X_test)
import matplotlib.pyplot as plt 
def plot(n,X_t,decode_X):
    plt.figure(figsize=(20,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(X_t[i].reshape(28,28))
        plt.gray()
        plt.axis('off')
        
        plt.subplot(2,n,i+1+n)
        plt.imshow(decode_X[i].reshape(28,28))
        plt.gray()
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot(10,X_test,deosde_imgs)

## ---(Sat Jun  2 17:25:19 2018)---
from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt 

def plot(n,X_t,decode_X):
    plt.figure(figsize=(20,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(X_t[i].reshape(28,28))
        plt.gray()
        plt.axis('off')
        
        plt.subplot(2,n,i+1+n)
        plt.imshow(decode_X[i].reshape(28,28))
        plt.gray()
        plt.axis('off')
    plt.tight_layout()
    plt.show()


(X_train,_),(X_test,_)=mnist.load_data()
X_train=X_train.astype('float32') / 255.
X_test=X_test.astype('float32') / 255.
print(X_test.shape)
X_test=X_test.reshape(X_test.shape[0], -1)
X_test.shape
X_train=X_train.reshape(X_train.shape[0],-1)
X_train.shape

input_size=784
encoding_size=64
batchsize=200
epochs=10
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(X_train,X_train,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
deosde_imgs=autoencoder.predict(X_test)
plot(10,X_test,deosde_imgs)
encoder=Model(input_image,encoded)
tmp=encoder.predict(X_test)
from keras.datasets import fashion_mnist
(x_train, _), (x_test, _) = fashion_mnist.load_data()
print(x_train[:10])
from keras.preprocessing import image
import numpy as np
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\test\\1\\video_1\\image_34.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()
encoding_size=64
batchsize=200
epochs=10
from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
e
input_image=Input(shape=(150,150,3))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(img_tensor,img_tensor,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\test\\1\\video_1\\image_34.jpg"
img = image.load_img(img_path, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.
# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
plt.imshow(img_tensor[0])
input_image=Input(shape=(28,28,3))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(img_tensor,img_tensor,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
input_size=(28,28,3)
input_image=Input(shape=input_size)
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
input_image=Input(shape=input_size)
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
input_size=2352
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(img_tensor,img_tensor,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
(X_train,_),(X_test,_)=mnist.load_data()
X_train.shape
X_train=X_train.reshape(X_train.shape[0],-1)
X_train.shape
print(img_tensor.shape)
img_tensor=img_tensor.reshape(img_tensor.shape[0],-1)
print(img_tensor.shape)
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(img_tensor,img_tensor,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
os.listdir(img_path)
import os
os.listdir(img_path)
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\test\\1"
os.listdir(img_path)
list(os.listdir(img_path))
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\test\\1\\video_1"
list(os.listdir(img_path))
len(os.listdir(img_path))
list1=[]
for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)
    
    
    img = image.load_img(img_path1, target_size=(300, 300))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.
    # Its shape is (1, 150, 150, 3)
    list1.append(img_tensor)

np.reshape(list1,(286,270000))
import pandas as pd
np.savetxt('features1.csv',np.reshape(list1,(286,270000)),delimiter=',')
x=pd.read_csv('features1.csv',header=None)
from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
input_img = Input(shape=(28, 28, 3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
print(autoencoder.summary())
autoencoder.fit(img_path, img_path, epochs=epochs, batch_size=batchsize, shuffle=True, validation_split=0.2, callbacks=[save_weights])
x.shape
print(x.shape)
x=pd.read_csv('features1.csv',header=None)
print(x.shape)
X_train.shape
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
input_size=270000
encoding_size=64
batchsize=200
epochs=10
img_tensor=img_tensor.reshape(img_tensor.shape[0],-1)
print(img_tensor.shape)
input_image=Input(shape=(input_size,))
encoded=Dense(encoding_size)(input_image)
decoded=Dense(input_size,activation='sigmoid')(encoded)
autoencoder=Model(input_image,decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(x,x,epochs=epochs,batch_size=batchsize,shuffle=True,validation_split=0.2)
deosde_imgs=autoencoder.predict(x)
plot(10,x,deosde_imgs)
encoder=Model(input_image,encoded)
tmp=encoder.predict(x)
x = x.reshape((x.shape[0], -1))
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import cv2
import sys
import os
from random import shuffle, randint, choice
def getRandomImage():
	imageSize = 100
	size = 25
	nbShapes = 5
 
	xy = lambda: randint(0,100)
 
	# Create a white image
	img = np.zeros((imageSize,imageSize,3), np.uint8)
	cv2.rectangle(img,(0,0),(imageSize,imageSize),(122,122,122) ,-1)
 
	greyImg = np.copy(img)
 
	# Adds some shapes
	for i in range(nbShapes):
		x0, y0 = xy(), xy()
		isRect = choice((True,False))
		if isRect:
			cv2.rectangle(img,(x0,y0),(x0+size,y0+size),(255,0,0) ,-1)
			cv2.rectangle(greyImg,(x0,y0),(x0+size,y0+size),(255,255,255) ,-1)
		else:
			cv2.circle(img,(x0,y0), size/2, (0,0,255), -1)
			cv2.circle(greyImg,(x0,y0), size/2, (255,255,255), -1)
 
	return cv2.resize(img,(48,48)), cv2.resize(greyImg,(48,48))

def getDataset(display=False):
	# Show what the dataset looks like
	if display:
		colorImg, greyImg = getRandomImage()
		img = np.hstack((colorImg, greyImg))
		cv2.imshow("Dataset",cv2.resize(img,(200,100)))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
 
	#for i in range
 
	x_train, x_test, y_train, y_test = [], [], [], []
 
	# Add training examples
	for i in range(10000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_train.append(greyImg.astype('float32')/255.)
		y_train.append(colorImg.astype('float32')/255.)
 
	# Add test examples
	for i in range(1000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_test.append(greyImg.astype('float32')/255.)
		y_test.append(colorImg.astype('float32')/255.)
 
	# Reshape
	x_train = np.array(x_train).reshape((-1,48,48,1))
	x_test = np.array(x_test).reshape((-1,48,48,1))
	y_train = np.array(y_train).reshape((-1,48,48,3))
	y_test = np.array(y_test).reshape((-1,48,48,3))
 
	return x_train, y_train, x_test, y_test

def getModel():
	input_img = Input(shape=(48, 48, 1))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	#6x6x32 -- bottleneck
	x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	x = UpSampling2D((2, 2), dim_ordering='tf')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
 
	#Create model
	autoencoder = Model(input_img, decoded)
	return autoencoder

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import cv2
import sys
import os
from random import shuffle, randint, choice

def getRandomImage():
	imageSize = 100
	size = 25
	nbShapes = 5
 
	xy = lambda: randint(0,100)
 
	# Create a white image
	img = np.zeros((imageSize,imageSize,3), np.uint8)
	cv2.rectangle(img,(0,0),(imageSize,imageSize),(122,122,122) ,-1)
 
	greyImg = np.copy(img)
 
	# Adds some shapes
	for i in range(nbShapes):
		x0, y0 = xy(), xy()
		isRect = choice((True,False))
		if isRect:
			cv2.rectangle(img,(x0,y0),(x0+size,y0+size),(255,0,0) ,-1)
			cv2.rectangle(greyImg,(x0,y0),(x0+size,y0+size),(255,255,255) ,-1)
		else:
			cv2.circle(img,(x0,y0), size/2, (0,0,255), -1)
			cv2.circle(greyImg,(x0,y0), size/2, (255,255,255), -1)
 
	return cv2.resize(img,(48,48)), cv2.resize(greyImg,(48,48))

def getDataset(display=False):
	# Show what the dataset looks like
	if display:
		colorImg, greyImg = getRandomImage()
		img = np.hstack((colorImg, greyImg))
		cv2.imshow("Dataset",cv2.resize(img,(200,100)))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
 
	#for i in range
 
	x_train, x_test, y_train, y_test = [], [], [], []
 
	# Add training examples
	for i in range(10000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_train.append(greyImg.astype('float32')/255.)
		y_train.append(colorImg.astype('float32')/255.)
 
	# Add test examples
	for i in range(1000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_test.append(greyImg.astype('float32')/255.)
		y_test.append(colorImg.astype('float32')/255.)
 
	# Reshape
	x_train = np.array(x_train).reshape((-1,48,48,1))
	x_test = np.array(x_test).reshape((-1,48,48,1))
	y_train = np.array(y_train).reshape((-1,48,48,3))
	y_test = np.array(y_test).reshape((-1,48,48,3))
 
	return x_train, y_train, x_test, y_test

def getModel():
	input_img = Input(shape=(48, 48, 1))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	#6x6x32 -- bottleneck
	x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	x = UpSampling2D((2, 2), dim_ordering='tf')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
 
	#Create model
	autoencoder = Model(input_img, decoded)
	return autoencoder

def trainModel():
	# Load dataset
	print("Loading dataset...")
	x_train_gray, x_train, x_test_gray, x_test = getDataset()
 
	# Create model description
	print("Creating model...")
	model = getModel()
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
 
	# Train model
	print("Training model...")
	model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
 
	# Evaluate loaded model on test data
	print("Evaluating model...")
	score = model.evaluate(x_train_gray, x_train, verbose=0)
	print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
 
	# Serialize model to JSON
	print("Saving model...")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
 
	# Serialize weights to HDF5
	print("Saving weights...")
	model.save_weights("model.h5")

def trainModel():
	# Load dataset
	print("Loading dataset...")
	x_train_gray, x_train, x_test_gray, x_test = getDataset()
 
	# Create model description
	print("Creating model...")
	model = getModel()
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
 
	# Train model
	print("Training model...")
	model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
 
	# Evaluate loaded model on test data
	print("Evaluating model...")
	score = model.evaluate(x_train_gray, x_train, verbose=0)
	print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
 
	# Serialize model to JSON
	print("Saving model...")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
 
	# Serialize weights to HDF5
	print("Saving weights...")
	model.save_weights("model.h5")

def testModel():
	# Load JSON model description
	with open('model.json', 'r') as json_file:
		modelJSON = json_file.read()
 
	# Build model from JSON description
	print("Loading model...")
	model = model_from_json(modelJSON)
 
	# Load weights
	print("Loading weights...")
	model.load_weights("model.h5")
 
	_, _, x_test_gray, x_test = getDataset()
	x_test_gray = x_test_gray[:10]
	x_test = x_test[:10]
 
	print("Making predictions...")
	predictions = model.predict(x_test_gray)
	x_test_gray = [cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) for img in x_test_gray]
 
	img = np.vstack((np.hstack(x_test_gray),np.hstack(predictions),np.hstack(x_test)))
 
	cv2.imshow("Input - Reconstructed - Ground truth",cv2.resize(img,(img.shape[1],img.shape[0])))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	arg = sys.argv[1] if len(sys.argv) == 2 else None 
	if arg is None:
		print ("Need argument")
	elif arg == "train":
		trainModel()
	elif arg == "test":
		testModel()
	elif arg == "dataset":
		getDataset(True)
	else:
		print ("Wrong argument")

def trainModel():
	# Load dataset
	print("Loading dataset...")
	x_train_gray, x_train, x_test_gray, x_test = getDataset()
 
	# Create model description
	print("Creating model...")
	model = getModel()
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
 
	# Train model
	print("Training model...")
	model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
 
	# Evaluate loaded model on test data
	print("Evaluating model...")
	score = model.evaluate(x_train_gray, x_train, verbose=0)
	print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
 
	# Serialize model to JSON
	print("Saving model...")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
 
	# Serialize weights to HDF5
	print("Saving weights...")
	model.save_weights("model.h5")

print("Loading dataset...")
x_train_gray, x_train, x_test_gray, x_test = getDataset()
from __future__ import print_function
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
# For reproducibility
np.random.seed(1000)
if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    # Create the model
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=250,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
    
    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
    
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
def plot(n, X, Decoded_X):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
        
        # reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

(X_train, _), (X_test, _) = mnist.load_data()
print(X_train.shape)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(X_train.shape)
X_train = X_train[..., 3]
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
input_size = 784
epochs = 5
batch_size = 256
input_img = Input(shape=(28, 28, 1)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])
(X_train, _), (X_test, _) = mnist.load_data()
print(X_train.shape)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])
import cv2
import numpy as np
img=cv2.imread(im_path, cv2.IMREAD_COLOR)
im_path='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Data\\Box\\image_7'
img=cv2.imread(im_path, cv2.IMREAD_COLOR)
img.shape
img.dtype
img=cv2.imread(im_path, cv2.IMREAD_COLOR)
img
img.dtype
im_path='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Data\\Box\\image_7.jpg'
img=cv2.imread(im_path, cv2.IMREAD_COLOR)
img.dtype
img.shape
img=cv2.resize(img,(300,300), interpolation = cv2.INTER_AREA)
img.shape
autoencoder.fit(img, img, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])
img=img[...,np.newaxis]
img.shape
img=img[np.newaxis,...]
img.shape
im_path='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Data\\Box\\image_7.jpg'
img=cv2.imread(im_path, cv2.IMREAD_COLOR)
img.shape
img=cv2.resize(img,(300,300), interpolation = cv2.INTER_AREA)
img=img[np.newaxis,...]
img.shape
autoencoder.fit(img, img, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])
input_img = Input(shape=(300, 300, 3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
autoencoder.fit(img, img, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])
im_path='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Data\\Box'
out_i=[]
im_path='C:\\Users\\nnidamanuru\\Downloads\\GEA\\Data\\Box'
out_i=[]
import os
for i in list(os.listdir(im_path)):
    img_path1=os.path.join(im_path,i)
    img=cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img=cv2.resize(img,(300,300), interpolation = cv2.INTER_AREA)
    out_i.append(img)

print(len(out_i))
myarray = np.asarray(out_i)
myarray
X_train=myarray
X_train = X_train.astype('float32') / 255.
print(X_train.shape)
epochs = 5
batch_size = 256
input_img = Input(shape=(300, 300, 3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
batch_size = 256
epochs = 5
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])

## ---(Tue Jun  5 18:06:43 2018)---
import cv2
img_path='C:\\Users\\nnidamanuru\\Pictures\\box1.jpg'
img = cv2.imread(img_path)
print(img.shape)
scale_percent1 = 88.3
height = int(img.shape[0] * scale_percent1 / 100)
scale_percent2 = 150
width = int(img.shape[1] * scale_percent2 / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path,resized)
img_path='C:\\Users\\nnidamanuru\\Pictures\\box2.jpg'
img = cv2.imread(img_path)
print(img.shape)
scale_percent1 = 88.3
height = int(img.shape[0] * scale_percent1 / 100)
scale_percent2 = 150
width = int(img.shape[1] * scale_percent2 / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)

cv2.imshow("Resized image", resized)
cv2.imwrite(img_path,resized)
deosde_imgs=autoencoder.predict(X_train)
runfile('C:/Users/nnidamanuru/autoencodes_convolution.py', wdir='C:/Users/nnidamanuru')
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

def plot(n, X, Decoded_X):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
        
        # reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_size = 784
epochs = 5
batch_size = 256

input_img = Input(shape=(28, 28, 1)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])

decoded_imgs = autoencoder.predict(X_test)
plot(10, X_test, decoded_imgs)

encoder = Model(input_img, encoded)
tmp = encoder.predict(X_test)
from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

list1=[]
img_path ="C:\\Users\\nnidamanuru\\Pictures\\Saved Pictures"
print(list(os.listdir(img_path)))
for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(300,300), interpolation = cv2.INTER_AREA)
    list1.append(image)

img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"
print(list(os.listdir(img_path)))
for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(300,300), interpolation = cv2.INTER_AREA)
    list1.append(image)

myarray = np.asarray(list1)
print(myarray.shape)
input_img = Input(shape=(300,300,3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model_auto.h5', monitor='val_loss', save_best_only=True)
print(autoencoder.summary())
X_train=myarray
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
decoded_imgs=autoencoder.predict(X_train)
def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


plot(10, X_train, decoded_imgs) 
from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint


def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

list1=[]
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"
print(list(os.listdir(img_path)))
for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(300,300), interpolation = cv2.INTER_AREA)
    list1.append(image)

myarray = np.asarray(list1)
print(myarray.shape)
nput_img = Input(shape=(300,300,3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model_auto.h5', monitor='val_loss', save_best_only=True)
print(autoencoder.summary())
X_train=myarray
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
decoded_imgs=autoencoder.predict(X_train)
plot(10, X_train, decoded_imgs) 
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
decoded_imgs=autoencoder.predict(X_train)
plot(10, X_train, decoded_imgs) 
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
decoded_imgs=autoencoder.predict(X_train)
plot(10, X_train, decoded_imgs) 
from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint


def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(300, 300,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()




list1=[]

img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"

print(list(os.listdir(img_path)))

for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(300,300), interpolation = cv2.INTER_AREA)
    list1.append(image)


myarray = np.asarray(list1)
print(myarray.shape)

input_img = Input(shape=(300,300,3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model_auto.h5', monitor='val_loss', save_best_only=True)

print(autoencoder.summary())
X_train=myarray
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
decoded_imgs=autoencoder.predict(X_train)
plot(10, X_train, decoded_imgs) 
plot(10, X_train, decoded_imgs) 
from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(500, 500,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(500, 500,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()




list1=[]

img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"

print(list(os.listdir(img_path)))

for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(500,500), interpolation = cv2.INTER_AREA)
    list1.append(image)


myarray = np.asarray(list1)
print(myarray.shape)

input_img = Input(shape=(500,500,3)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model_auto.h5', monitor='val_loss', save_best_only=True)

print(autoencoder.summary())
X_train=myarray

autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])

decoded_imgs=autoencoder.predict(X_train)
plot(10, X_train, decoded_imgs) 
print('hello world')
runfile('C:/Users/nnidamanuru/ne1.py', wdir='C:/Users/nnidamanuru')
def sum1(a,b):
    c=a+b
    return c

print(sum1(2,5))
sum1(2,5)

## ---(Thu Jun  7 13:19:56 2018)---
import cv2

## ---(Thu Jun  7 13:21:37 2018)---
import cv2
img_path='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_1.jpg'
img = cv2.imread(img_path)
img_path='C:\\Users\\nnidamanuru\\Pictures\\imt2.jpg'
img = cv2.imread(img_path)
print(img.shape)
img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_1.jpg'
img1 = cv2.imread(img_path1)
print(img1.shape)
print(img.shape)
resized = cv2.resize(img1, (575,411), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)
img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_2.jpg'
img1 = cv2.imread(img_path1)
print(img1.shape)
resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)
img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_3.jpg'
img1 = cv2.imread(img_path1)
print(img1.shape)
resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)
img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_4.jpg'
img1 = cv2.imread(img_path1)
print(img1.shape)
resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)
img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_5.jpg'
img1 = cv2.imread(img_path1)
print(img1.shape)
resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)

## ---(Wed Jun 13 18:04:48 2018)---
from keras.models import Model
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
def plot(n, X, Decoded_X):
    plt.figure(figsize=(30, 30))
    for i in range(n):
        print('original')
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(500, 500,3))
        plt.colors()
        plt.axis('off')
        
        print('reconstruction')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(500, 500,3))
        plt.colors()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

list1=[]

img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"

print(list(os.listdir(img_path)))

for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.resize(image,(700,700), interpolation = cv2.INTER_AREA)
    list1.append(image)

myarray = np.asarray(list1)
print(myarray.shape)
input_img = Input(shape=(700,700,3)) 
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoded
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model_auto_r.h5', monitor='val_loss', save_best_only=True)
print(autoencoder.summary())
X_train=myarray
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
autoencoder.fit(X_train, X_train, epochs=1, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[save_weights])
from PIL import Image
import cv2
import os
x1 = 732
y1 = 168
x2 = 1405
y2 = 1053
path2='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_images'
path4='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_cropped_scenario1'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 657
y1 = 109
x2 = 1431
y2 = 1078

path2='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_images'
path4='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_cropped_scenario2'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077

path2='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_images'
path4='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_cropped_scenario2'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 484
y1 = 314
x2 = 1063
y2 = 1077

path2='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_images'
path4='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_cropped_scenario3'
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

x1 = 445
y1 = 253
x2 = 1063
y2 = 1077
path2='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_images'
path4='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\Fullbox_cropped_scenario4'

for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
    
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


## ---(Tue Jun 19 14:28:16 2018)---
from keras.layers import Input, Dense, LeakyReLU, Activation
from keras.models import Model
from keras.datasets import mnist 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import math
import keras

print(keras.__version__)
def plot_images(images):
    plt.figure(figsize=(20, 8))
    num_images = images.shape[0]
    rows = int(math.sqrt(images.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = deprocess(images[i])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def build_generator(input):
   x =  Dense(128)(input)
   x = LeakyReLU(alpha=0.01)(x)
   x = Dense(784)(x)
   x = Activation('tanh')(x)
   generator = Model(input, x)
   return generator

def test_generator(generator, n_samples, sample_size):
    latent_samples = make_latent_samples(n_samples, sample_size)
    images = generator.predict(latent_samples)
    plot_images(images)

def build_descriminator(descriminator_input_size):
   input = Input(shape=(descriminator_input_size,))     
   x =  Dense(128)(input)
   x = LeakyReLU(alpha=0.01)(x)
   x = Dense(1)(x)
   x = Activation('sigmoid')(x)
   descriminator = Model(input, x)
   return descriminator 

def build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate):
    input = Input(shape=(latent_sample_size,))     
    
    #build generator model
    generator = build_generator(input)
    print(generator.summary())
    
    #build descriminator model
    descriminator =  build_descriminator(descriminator_input_size)
    print(descriminator.summary())
    
    #build adversarial model = generator + discriminator
    gan = Model(input, descriminator(generator(input)))
    print(gan.summary())
    
    descriminator.compile(optimizer=Adam(lr=d_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return generator, descriminator, gan

def make_latent_samples(n_samples, sample_size):
    return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    #return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


def train_gan(generator, descriminator, gan, X_train_real, batch_size, epochs, latent_sample_size):
    y_train_real, y_train_fake = make_labels(batch_size)
    
    losses = []
    nbatches = len(X_train_real) // batch_size
    for e in range(epochs):
        for i in range(nbatches):
            # real MNIST digit images
            X_batch_real = X_train_real[i*batch_size:(i+1)*batch_size]
            
            # latent samples and the generated digit images
            latent_samples = make_latent_samples(batch_size, latent_sample_size)
            X_batch_fake = generator.predict_on_batch(latent_samples)
            
            # train the discriminator to detect real and fake images
            X = np.concatenate((X_batch_real, X_batch_fake))
            y = np.concatenate((y_train_real, y_train_fake))
            descriminator.trainable = True
            metrics = descriminator.train_on_batch(X, y)
            d_loss = metrics[0]
            d_acc = metrics[1]
            log = "%d/%d: [discriminator loss: %f, acc: %f]" % (e, i, d_loss, d_acc)
            
            # train gan with latent_samples and y_train_real of 1s
            descriminator.trainable = False
            metrics = gan.train_on_batch(latent_samples, y_train_real)
            
            g_loss = metrics[0]
            g_acc = metrics[1]
            log = "%s [adversarial loss: %f, acc: %f]" % (log, g_loss, g_acc)
            print(log)
            losses.append((d_loss, g_loss, d_acc, g_acc))
    
    return losses

def plot_loss(losses):
   losses = np.array(losses)
   fig, ax = plt.subplots()
   plt.plot(losses.T[0], label='Discriminator')
   plt.plot(losses.T[1], label='GAN')
   plt.title("Train Losses")
   plt.legend()
   plt.show() 

def preprocess(x):    
    x = x.reshape(-1, 784) # 784=28*28
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x


def deprocess(x):
    x = (x / 2 + 1) * 255
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    x = x.reshape(28, 28)
    return x       

batch_size = 64
epochs = 10
latent_sample_size = 100
g_learning_rate = 0.0001 
d_learning_rate = 0.001
descriminator_input_size = 784
import cv2
import os
list1=[]
img_path ="C:\\Users\\nnidamanuru\\Downloads\\GEA\\New folder"
print(list(os.listdir(img_path)))
for i in list(os.listdir(img_path)):
    img_path1=os.path.join(img_path,i)    
    image = cv2.imread(img_path1)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(28,28), interpolation = cv2.INTER_AREA)
    list1.append(image)

myarray = np.asarray(list1)
print(myarray.shape)
X_train=myarray
X_train=X_train.reshape(X_train.shape[0],-1)
X_train.shape
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
test_generator(generator, 10, latent_sample_size) 
batch_size = 64
epochs = 100
latent_sample_size = 100
g_learning_rate = 0.0001 
d_learning_rate = 0.001
descriminator_input_size = 784
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
test_generator(generator, 10, latent_sample_size) 
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
test_generator(generator, 10, latent_sample_size) 
latent_sample_size = 50
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
#generate images using trained model
test_generator(generator, 5, latent_sample_size) 
batch_size = 64
epochs = 500
latent_sample_size = 100
g_learning_rate = 0.0001 
d_learning_rate = 0.001
descriminator_input_size = 784
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
test_generator(generator, 5, latent_sample_size) 
test_generator(generator, 1, latent_sample_size) 
test_generator(generator, 1, latent_sample_size) 
test_generator(generator, 2, latent_sample_size) 
test_generator(generator, 1, latent_sample_size) 
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
test_generator(generator, 1, latent_sample_size) 
def plot(n,X_t):
    plt.figure(figsize=(30,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(X_t[i].reshape(28,28))
        plt.gray()
        plt.axis('off')
    
    
    plt.tight_layout()
    plt.show()

X_train.shape
plot(1,X_train)
plot(5,X_train)
def build_generator(input):
   x =  Dense(128)(input)
   x = LeakyReLU(alpha=0.01)(x)
   x =  Dense(256)(input)
   x = LeakyReLU(alpha=0.01)(x)
   x = Dense(784)(x)
   x = Activation('tanh')(x)
   generator = Model(input, x)
   return generator

generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)
test_generator(generator, 1, latent_sample_size) 

## ---(Wed Jun 20 14:47:40 2018)---
import math
from collections import Counter
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in common])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

text1 = 'This is an article on analytics vidhya' 
text2 = 'article on analytics vidhya is about natural language processing'
vector1 = text_to_vector(text1) 
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
text1 = 'How are you' 
text2 = 'How do you  do'
vector1 = text_to_vector(text1) 
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
text1 = 'He is Good' 
text2 = 'He is Nice'
vector1 = text_to_vector(text1) 
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

cosine = get_cosine(vector1, vector2)
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(x)
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

cosine = get_cosine(vector1, vector2)
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

cosine = get_cosine(vector1, vector2)
print(vector1['He'])
print(vector1['is'])
print(vector1['Good'])
print(vector1[He])
print(vector1['Neh'])
print(vector1['good'])
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

cosine = get_cosine(vector1, vector2)
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    print(denominator)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

cosine = get_cosine(vector1, vector2)
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 
    return distances[-1]

print(levenshtein("analyze","analyse"))
print(levenshtein("analyze","anajyse"))
print(levenshtein("analyze","an4jyse"))
print(levenshtein("analyze","an4jyse4"))
print(levenshtein("analyze","adklfnsfgj"))
print(levenshtein("analyze","aaaaaa"))
print(levenshtein("analyze","bagaaaa"))
print(levenshtein("analyze",""))
print(levenshtein("analyze","g"))
print(levenshtein("analyze","gacdsed"))
if len(s1) > len(s2):
    s1,s2 = s2,s1 
    print(s1,s2)

distances = range(len(s1) + 1) 
for index2,char2 in enumerate(s2):
    newDistances = [index2+1]
    for index1,char1 in enumerate(s1):
        if char1 == char2:
            newDistances.append(distances[index1]) 
        else:
             newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
    distances = newDistances 

return distances[-1]
print(levenshtein("analyze","gacdsed"))
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
        print(s1,s2)
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 
    return distances[-1]

print(levenshtein("analyze","gacdsed"))
levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
        print(s1,s2)
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 

levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
        print(s1)
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 


levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1)
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 

levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 

levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 

levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
                print(newDistances)
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 

levenshtein("analyze","gacdsed")
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
                print(newDistances)
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances
        return distances

def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
                print(newDistances)
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances
        return distances

levenshtein("analyze","gvfcdgd")
return distances[-1]
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
                print(newDistances)
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances
    return distances[-1]

levenshtein("analyze","gvfcdgd")
levenshtein("analyze","gvfcded")
levenshtein("analyze","gvfcdad")
levenshtein("analyze","avfcdad")
levenshtein("analyze","avacdad")
levenshtein("analyze","anacdad")
levenshtein("analyze","analyse")

## ---(Thu Jun 28 17:47:48 2018)---
import math
from collections import Counter
ef get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    print(denominator)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print(numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    print(denominator)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator


def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

text1 = 'He is Good' 
text2 = 'He is Nice'
vector1 = text_to_vector(text1) 
print(vector1['good'])
cosine = get_cosine(vector1, vector2)
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
text1 = 'How are you' 
text2 = 'How you are'
vector1 = text_to_vector(text1) 
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
print(vector1)
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print('numerator:',numerator)
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    print('denominator:',denominator)
    
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator


def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

cosine = get_cosine(vector1, vector2)
cosine
text2 = 'where you are'
cosine = get_cosine(vector1, vector2)
cosine
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)
cosine
text1 = 'How are you' 
text2 = 'where you are'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
vector1
vector2
common = set(vec1.keys()) & set(vec2.keys())
common = set(vector1.keys()) & set(vector2.keys())
common
print(len(common))
vector2.keys()
text1 = 'Hi There I am working remotely and I cannot connect to VPN as it says my\
username and/or password is incorrect. I am using the same login details for computer and server' 
text2 = 'where you are'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
common = set(vector1.keys()) & set(vector2.keys())
print(len(common))
text1 = 'Hi There I am working remotely and I cannot connect to VPN as it says my\
text1 = 'Hi There I am working remotely and I cannot connect to VPN as it says my\
username and/or password is incorrect. I am using the same login details for computer and server' 
text2 = 'where you or'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
common = set(vector1.keys()) & set(vector2.keys())
print(len(common))
text1 = 'Hi There I am working remotely and I cannot connect to VPN as it says my\
username and/or password is incorrect. I am using the same login details for computer and server' 
text2 = 'where you is'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
common = set(vector1.keys()) & set(vector2.keys())
print(len(common))
vector1
dissimilar=set(vector1.keys()) ^ set(vector2.keys())
print(dissimilar)

## ---(Thu Jul  5 13:01:04 2018)---
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize   
import nltk
import re
from collections import Counter
from stemming.porter2 import stem
from nltk.util import ngrams
def Sentence_clean(text):
     sentence_clean= re.sub("[^a-zA-Z]"," ", text)
     sentence_clean = sentence_clean.lower()
     tokens = word_tokenize(sentence_clean)
     stop_words = set(stopwords.words("english"))
     sentence_clean_words = [w for w in tokens if not w in stop_words]
     return ' '.join(sentence_clean_words)

def unigram_n(text): 
    words = text.split() 
    return Counter(words)

def to_do_stem(text1):
    s=''
    for word in list(text1.split(" ")):
        s+=' '+(str(stem(word)))
    return s

def bigram_n(text):
    bigram = ngrams(text.split(), n=2)
    neh=list(bigram)
    s=[]
    for i in list(range(len(neh))):
        s.append((neh[i][0]+' '+neh[i][1]))
    return Counter(s)

def trigram_n(text):
    bigram = ngrams(text.split(), n=3)
    neh=list(bigram)
    t=[]
    for i in list(range(len(neh))):
        t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
    return Counter(t)

def word_match_probability(len_vec1,num_questions,list_1):
    list_proba=[]
    for i in list(range(num_questions)):
        prob=list_1[i] / len_vec1
        list_proba.append(prob)
    return list_proba

train_data_sim = pd.read_csv('C:/Users/nnidamanuru/Documents/nielsen_test262.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv')
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize   
import nltk
import re
from collections import Counter
from stemming.porter2 import stem
from nltk.util import ngrams
def Sentence_clean(text):
     sentence_clean= re.sub("[^a-zA-Z]"," ", text)
     sentence_clean = sentence_clean.lower()
     tokens = word_tokenize(sentence_clean)
     stop_words = set(stopwords.words("english"))
     sentence_clean_words = [w for w in tokens if not w in stop_words]
     return ' '.join(sentence_clean_words)



# # Splitting the sentence by single word

def unigram_n(text): 
    words = text.split() 
    return Counter(words)



# # Stemming(root word) the each word

def to_do_stem(text1):
    s=''
    for word in list(text1.split(" ")):
        s+=' '+(str(stem(word)))
    return s




# # Splitting the sentence by two words

def bigram_n(text):
    bigram = ngrams(text.split(), n=2)
    neh=list(bigram)
    s=[]
    for i in list(range(len(neh))):
        s.append((neh[i][0]+' '+neh[i][1]))
    return Counter(s)



# # Splitting the sentence by three words

def trigram_n(text):
    bigram = ngrams(text.split(), n=3)
    neh=list(bigram)
    t=[]
    for i in list(range(len(neh))):
        t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
    return Counter(t)



# # Getting the probability for the Sentence with other
def word_match_probability(len_vec1,num_questions,list_1):
    list_proba=[]
    for i in list(range(num_questions)):
        prob=list_1[i] / len_vec1
        list_proba.append(prob)
    return list_proba

train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv', error_bad_lines=False)
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Downloads\\train.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_test262.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Downloads\\test_neh.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Downloads\\test_neh')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Downloads\\test_neh.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Downloads\\Copy of ChatBot Questions%2F Response.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\Copy of ChatBot Questions%2F Response.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\neilsen_test1.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\neilsen_test1.csv').a.encode('utf-8').strip()
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\test_neh.csv').a.encode('utf-8').strip()
train_data_sim = pd.read_csv('https://vsoftconsultinggroup-my.sharepoint.com//personal//miragavarapu_vsoftconsulting_com//Documents//BizSol//Practices//Emerging%20Tech//Shared//Nielsen%20Incident%20Data%20-%20Part%201.xlsx?web=1')
import urllib.request
train_data_sim = urllib.request.urlopen('https://vsoftconsultinggroup-my.sharepoint.com//personal//miragavarapu_vsoftconsulting_com//Documents//BizSol//Practices//Emerging%20Tech//Shared//Nielsen%20Incident%20Data%20-%20Part%201.xlsx?web=1')
train_data_sim = pd.read_csv('https://vsoftconsultinggroup-my.sharepoint.com//personal//miragavarapu_vsoftconsulting_com//Documents//BizSol//Practices//Emerging%20Tech//Shared//Nielsen%20Incident%20Data%20-%20Part%201.xlsx?web=1').encode('utf-8')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\test_similarity.csv')
train_data_sim = pd.read_csv('C:\Users\nnidamanuru\Documents\nielsen_neh1.csv')
train_data_sim = pd.read_csv(u'C:\Users\nnidamanuru\Documents\nielsen_neh1.csv')
train_data_sim = pd.read_csv(u'C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
print(train_data_sim.shape)
print(train_data_sim.info())
question_summary=train_data_sim['Issue Summary']
question_Description=train_data_sim['Issue Description']
nltk.download('punkt')
nltk.download('stopwords')
clean_question_summary=list(map(Sentence_clean,question_summary))
clean_question_Description=list(map(Sentence_clean,question_Description))
print(clean_question_Description[6])
stem_question_summary=list(map(to_do_stem,clean_question_summary))
stem_question_Description=list(map(to_do_stem,clean_question_Description))
print(stem_question_Description[6])
from flask_wtf import Form
from wtforms.fields import *
from wtforms.validators import Required, Email
from flask_wtf import Form
from wtforms.fields import *
from wtforms.validators import Required, Email

class SignupForm(Form):
    name = TextField(u'Your name', validators=[Required()])
    password = TextField(u'Your favorite password', validators=[Required()])
    email = TextField(u'Your email address', validators=[Email()])
    birthday = DateField(u'Your birthday')
    
    a_float = FloatField(u'A floating point number')
    a_decimal = DecimalField(u'Another floating point number')
    a_integer = IntegerField(u'An integer')
    
    now = DateTimeField(u'Current time',
                        description='...for no particular reason')
    sample_file = FileField(u'Your favorite file')
    eula = BooleanField(u'I did not read the terms and conditions',
                        validators=[Required('You must agree to not agree!')])
    
    submit = SubmitField(u'Signup')

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
os.chdir('C:\\Users\\nnidamanuru')
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
os.chdir('C:\\Users\\nnidamanuru')
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
os.chdir('C:\\Users\\nnidamanuru')
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app

app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'
# print(app.config['IMAGE_DIRECTORY'])
print(app.config['MONILIA_SLICES_DIRECTORY'])
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
    list_proba=[]
    for i in list(range(num_questions)):
        prob=list_1[i] / len_vec1
        list_proba.append(prob)
    return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        text=text1
        def n_gram_nielsen(text):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        text=text1
        def n_gram_nielsen(text):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')








def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        text=text1
        def n_gram_nielsen(text):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        text=text1
        def n_gram_nielsen(text):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')








def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')













print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                print(vector1_uni)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')


print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))










def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')


print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))










def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            print("after description",text1)
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("before summary",text1)
        
        def n_gram_nielsen(text1):
            print("after summary",text1)
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')

print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            print("after description",text1)
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("before summary",text1)
        
        def n_gram_nielsen(text1):
            print("after summary",text1)
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')
print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
    
    print(clean_question_Description[6])
    
    
    # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))
    
    
    print(stem_question_Description[6])



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_des1=[]
            list_des2=[]
            list_des3=[]
            for i in list(range(len(stem_question_Description))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_Description[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_des1.append(len(common_uni))
                list_des2.append(len(common_bi))
                list_des3.append(len(common_tri))
            return list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri
        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("Entered into summary")
        
        def n_gram_nielsen(text1):
            #unigram common count in Description
            list_summ1=[]
            list_summ2=[]
            list_summ3=[]
            for i in list(range(len(stem_question_summary))):
                clean_text1=Sentence_clean(text1)
                stem_text1=to_do_stem(clean_text1)
                text2 = stem_question_summary[i] 
                vector1_uni = unigram_n(stem_text1)
                vector2_uni = unigram_n(text2)
                vector1_bi = bigram_n(stem_text1)
                vector2_bi = bigram_n(text2)
                vector1_tri = trigram_n(stem_text1)
                vector2_tri = trigram_n(text2)
                common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
                common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
                common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
                list_summ1.append(len(common_uni))
                list_summ2.append(len(common_bi))
                list_summ3.append(len(common_tri))
            return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')
print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))

## ---(Fri Jul  6 11:34:59 2018)---
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    def n_gram_nielsen(text1):
        #unigram common count in Description
        list_summ1=[]
        list_summ2=[]
        list_summ3=[]
        for i in list(range(len(stem_question_summary))):
            clean_text1=Sentence_clean(text1)
            stem_text1=to_do_stem(clean_text1)
            text2 = stem_question_summary[i] 
            vector1_uni = unigram_n(stem_text1)
            vector2_uni = unigram_n(text2)
            vector1_bi = bigram_n(stem_text1)
            vector2_bi = bigram_n(text2)
            vector1_tri = trigram_n(stem_text1)
            vector2_tri = trigram_n(text2)
            common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
            common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
            common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
            list_summ1.append(len(common_uni))
            list_summ2.append(len(common_bi))
            list_summ3.append(len(common_tri))
        return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
     
     # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))   



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        
        
        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("Entered into summary")
        
        
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')
print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='description'))
print(similarity_sentence(text1="We were requested to get the access for unix servers",validate='summary'))
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
os.chdir('C:\\Users\\nnidamanuru')
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
os.chdir('E:\\nielsen_ml')
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

import socket
from socket import socket
from wtf.app import app as application
from wtf import *
results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
results = nlp_nielsen_similarity.similarity_sentence(text='We were requested to get the access for unix servers', text_type='summary')
results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', text_type='summary')
results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', validate='summary')
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', validate='summary')
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from wtf import *
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', validate='summary')
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

from wtf import *
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from wtf import *
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)


@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', validate='summary')
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

results = nlp_nielsen_similarity.similarity_sentence(text1='We were requested to get the access for unix servers', validate='summary')
results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
import jsonify
@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)

from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
import os
from flask import Flask, Response
from markupsafe import escape
#from forms import SignupForm
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    return jsonify(message)

import sys
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-text1")
parser.add_option("-e")
parser.add_option("-g")
print(sys.argv) # unmodified
(options, args) = parser.parse_args()
print (options, args)
sys.argv = ['nlp_neilsen_similarity.py','-e','afbdfgsdfg','-g','summary'] # define your commandline arguments here
(options, args) = parser.parse_args()
print (options, args)
print (args)
parser.add_option("-w")
parser.add_option("-te")
parser.add_option("--te")
parser.add_option("-----text1")
parser.add_option("--text1")
parser.add_option("--validate")

## ---(Fri Jul  6 19:16:41 2018)---
train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    def n_gram_nielsen(text1):
        #unigram common count in Description
        list_summ1=[]
        list_summ2=[]
        list_summ3=[]
        for i in list(range(len(stem_question_summary))):
            clean_text1=Sentence_clean(text1)
            stem_text1=to_do_stem(clean_text1)
            text2 = stem_question_summary[i] 
            vector1_uni = unigram_n(stem_text1)
            vector2_uni = unigram_n(text2)
            vector1_bi = bigram_n(stem_text1)
            vector2_bi = bigram_n(text2)
            vector1_tri = trigram_n(stem_text1)
            vector2_tri = trigram_n(text2)
            common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
            common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
            common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
            list_summ1.append(len(common_uni))
            list_summ2.append(len(common_bi))
            list_summ3.append(len(common_tri))
        return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
     
     # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))   



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        
        
        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        frresult = result.sort_values('score_uni',ascending=False)
        
        
        print(frresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("Entered into summary")
        
        
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return result.to_csv('clean_similarity_unibitri_proba_score_test1.csv')











import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
similarity_sentence('We were requested to get the access for unix servers', 'summary')
similarity_sentence(text='We were requested to get the access for unix servers', text-type='summary')
similarity_sentence(text='We were requested to get the access for unix servers', text_type='summary')
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(' We were requested to get the access for unix servers', 'summary')
            print(results)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

import os
os.chdir('C:\\Users\\nnidamanuru\\Downloads\\nielsen_ml')
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            print(results)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

import os
os.chdir('C:\\Users\\nnidamanuru\\Downloads\\nielsen_ml')
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            print(results)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)

import os
os.chdir('C:\\Users\\nnidamanuru\\Downloads\\nielsen_ml')
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
            print(results)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)

print(check_similarity)
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
            print(results)
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)

print(results)
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
print(results)
s=5
print(s)
import nlp_nielsen_similarity
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
print(result)
print(results)
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    def n_gram_nielsen(text1):
        #unigram common count in Description
        list_summ1=[]
        list_summ2=[]
        list_summ3=[]
        for i in list(range(len(stem_question_summary))):
            clean_text1=Sentence_clean(text1)
            stem_text1=to_do_stem(clean_text1)
            text2 = stem_question_summary[i] 
            vector1_uni = unigram_n(stem_text1)
            vector2_uni = unigram_n(text2)
            vector1_bi = bigram_n(stem_text1)
            vector2_bi = bigram_n(text2)
            vector1_tri = trigram_n(stem_text1)
            vector2_tri = trigram_n(text2)
            common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
            common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
            common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
            list_summ1.append(len(common_uni))
            list_summ2.append(len(common_bi))
            list_summ3.append(len(common_tri))
        return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
     
     # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))   



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        
        
        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        fresult = result.sort_values('score_uni',ascending=False)
        
        
        print(fresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("Entered into summary")
        
        
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return fresult.head(5)











import nlp_nielsen_similarity
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
print(results)
#print(fresult.head(5))
def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams
    
    
    ## cleaning the sentence by removing the stop words and other than Alphabets
    
    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)
    
    
    # # Splitting the sentence by single word
    
    def unigram_n(text): 
        words = text.split() 
        return Counter(words)
    
    
    # # Stemming(root word) the each word
    
    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s
    
    
    
    # # Splitting the sentence by two words
    
    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)
    
    
    # # Splitting the sentence by three words
    
    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)
    
    
    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    def n_gram_nielsen(text1):
        #unigram common count in Description
        list_summ1=[]
        list_summ2=[]
        list_summ3=[]
        for i in list(range(len(stem_question_summary))):
            clean_text1=Sentence_clean(text1)
            stem_text1=to_do_stem(clean_text1)
            text2 = stem_question_summary[i] 
            vector1_uni = unigram_n(stem_text1)
            vector2_uni = unigram_n(text2)
            vector1_bi = bigram_n(stem_text1)
            vector2_bi = bigram_n(text2)
            vector1_tri = trigram_n(stem_text1)
            vector2_tri = trigram_n(text2)
            common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
            common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
            common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
            list_summ1.append(len(common_uni))
            list_summ2.append(len(common_bi))
            list_summ3.append(len(common_tri))
        return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri
    
    
    # # Imporing the Train Data
    
    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')
    
    print(train_data_sim.shape)
    
    print(train_data_sim.info())
    
    
    # # Issue summary and description sentences are moved into the variables
    
    question_summary=train_data_sim['Issue Summary']
    
    question_Description=train_data_sim['Issue Description']
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    # # Cleaning the Sentences
    
    
    clean_question_summary=list(map(Sentence_clean,question_summary))
    
    clean_question_Description=list(map(Sentence_clean,question_Description))
     
     # # Stemming the words in Sentence
    
    stem_question_summary=list(map(to_do_stem,clean_question_summary))
    
    
    stem_question_Description=list(map(to_do_stem,clean_question_Description))   



# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        
        
        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        ## Getting the probability of matching a sentence 
        
        
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)
        
        
        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)
        
        
        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)
        
        
        
        print(len(output_uni))
        
        
        # # Making each list into DataFrame
        
        
        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)
        
        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)
        
        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']
        
        
        ## Sorting for the top 5 values
        
        fresult = result.sort_values('score_uni',ascending=False)
        
        
        #print(fresult.head(5))
    
    
    
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively
    
    elif validate=='summary':
        print("Entered into summary")
        
        
        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
        
        
        ## Getting the probability of matching a sentence
        
        
        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)
        
        
        ## Making each list into DataFrame
        
        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)
        
        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)
        
        
        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']
        
        
        ## Sorting for the top 5 values
        
        
        fresult = result.sort_values('score_uni_summ',ascending=False)
        
        #print(fresult.head(5))
    
    
    ## Write CSV file from DataFrame
    
    
    return fresult.head(5)











import nlp_nielsen_similarity
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
import nlp_nielsen_similarity
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
print(results)
import os
os.chdir('C:\\Users\\nnidamanuru\\Downloads\\nielsen_ml')
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
            print(nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary'))
            message ={"status":"success",
                "message": results}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)

import nlp_nielsen_similarity
results = nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary')
print(results)
print(nlp_nielsen_similarity.similarity_sentence("We were requested to get the access for unix servers", 'summary'))
import flask
from flask_bootstrap import Bootstrap
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask import Flask, Blueprint, redirect, url_for, render_template, flash, request, session, abort, make_response
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import time
from flask import Flask, Response
from markupsafe import escape
from flask_nav import Nav
from werkzeug import secure_filename
import tensorflow as tf
import sys
import os
import random
from os import listdir
import jsonify
import nlp_nielsen_similarity

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    app.debug = True
    Bootstrap(app)
    return app


app = create_app()
app.secret_key = '#@$%$^#$sed23fgs@#$@!#'

# print(app.config['IMAGE_DIRECTORY'])
#print(app.config['NIELSEN_SLICES_DIRECTORY'])



@app.route('/')
def index():
    message = {"message": "working!"}
    # return jsonify(message)
    return flask.jsonify(message)




@app.route('/check_similarity',methods = ['POST', 'GET'])
def check_similarity():
    error = None
    if request.method == 'POST':
        # use request.data or request.get_json()['text'] if request.form does not work
        # print(request.data) 
        # print(request.get_json()['text']) 
        if request.form['text'] != '' and request.form['text_type'] != '':
            text = request.form['text']
            text_type = request.form['text_type']
            results = nlp_nielsen_similarity.similarity_sentence(text, text_type)
            message ={"status":"success",
                "message": results.to_json()}
        else:
            message ={"status":"error",
                "message":"text or text_type can't be empty"}
    else:
        message ={"status":"error",
            "message":"Method not supported"}
    
    return flask.jsonify(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12954)

def word_match_probability(len_vec1,num_questions,list_1):
    list_proba=[]
    for i in list(range(num_questions)):
        if len_vec1 != 0:
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        elif len_vec1 == 0:
            list_proba.append('number of words zero')
    
    return list_proba

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, RepeatVector, GRU, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pandas as pd
label_length = 5
n_classes = 10
n_train_samples = 100
n_test_samples = 100
epochs = 10
batch_size = 32
def make_dataset(X, y, num_samples, length):
    height = 28
    width = 28*length
    samples = np.ndarray(shape=(num_samples, width, height), dtype=np.float32)
    labels = []
    permutation = np.random.permutation(X.shape[0])
    
    start = 0
    for i in range(num_samples):
        rand_indices = [permutation[index] for index in range(start, start + length)]
        sample = np.hstack([X[index] for index in rand_indices])
        label = [y[index] for index in rand_indices]
        start += length 
        if start >= len(permutation):
                permutation = np.random.permutation(X.shape[0])
                start = 0
        samples[i,:,:] = sample.T
        labels.append(label)
    return {"images": samples, "labels": np.array(labels)}

def show_img_label(train_data, i):
    img = train_data.get('images')[i]
    plt.imshow(img.T, cmap='gray')
    label = train_data.get('labels')[i]
    print(label)
    plt.show()


def prepare_images(img_data):   
    return np.expand_dims(img_data, -1).astype('float32')/255.0

(digits_images_train, digits_labels_train), (digits_images_test, digits_labels_test) = mnist.load_data()
train_data = make_dataset(digits_images_train, digits_labels_train, n_train_samples, label_length)
show_img_label(train_data, 10)
X_train = prepare_images(train_data.get('images'))
y_train = train_data.get('labels')
y_train = np.array([np_utils.to_categorical(i, num_classes=n_classes) for i in y_train])
#convolutional & max pool layers to extract features out of image    
input_data = Input(shape=X_train.shape[1:], name='the_input')
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(input_data)
inner = MaxPooling2D((2, 2), name='max1')(inner)
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(inner)
inner = MaxPooling2D((2, 2), name='max2')(inner)
feature_vector = Flatten()(inner)
inner = Dense(1024, activation='relu', name='dense1')(feature_vector)
inner = RepeatVector(label_length)(inner)
inner = Bidirectional(GRU(64, return_sequences=True, name='gru'))(inner)
output = TimeDistributed(Dense(n_classes, activation='softmax', name='dense2'))(inner)
model = Model(inputs = input_data, outputs = output)
print(model.summary())
for layer in model.layers:
    print(layer.name, layer.input.shape, layer.output.shape)


model.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
model.fit(x=X_train, y=y_train, epochs=epochs, 
                    batch_size=batch_size, validation_split=0.1,
                    callbacks=[save_weights])
test_data = make_dataset(digits_images_test, digits_labels_test, n_test_samples, label_length)
show_img_label(test_data, 1)
from random import randrange
from matplotlib import patches
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
img_width, img_height = 28, 28
nclasses = 36
epochs = 10
batchsize = 32
n_train_samples = 100