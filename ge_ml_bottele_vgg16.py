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
##Model3
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

##model1
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