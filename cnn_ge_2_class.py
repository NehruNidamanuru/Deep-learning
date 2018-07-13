# Importing modules
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
    class_mode='categorical')

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.001)

save_weights = ModelCheckpoint('model11705_2class.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=(nb_train_samples//batch_size)+1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=(nb_validation_samples//batch_size)+1,
    callbacks=[save_weights, reduce_lr])

historydf = pd.DataFrame(history.history, index=history.epoch)
plot_loss_accuracy(history)

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

from keras.models import model_from_json
model1 = model_from_json('Downloads\\model.json')
model1.load_weights('Downloads\\model11705_2clas.h5')

test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#print(test_generator.filenames)
probabilities = model1.predict_generator(test_generator, 26//5)


mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i]
    i += 1
#od = collections.OrderedDict(sorted(mapper.items()))    
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission1.csv', columns=['id','label'], index=False)
