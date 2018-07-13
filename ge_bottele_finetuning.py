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

top_model.compile(loss='categorical_crossentropy', 
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
##early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')  
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

test_dir='C:\\Users\\nnidamanuru\\Downloads\\Test_Ge'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#print(test_generator.filenames)
probabilities = model.predict_generator(test_generator, 26//5)

mapper = {}
i = 0
for file in test_generator.filenames:
    id = (file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1
#od = collections.OrderedDict(sorted(mapper.items()))    
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission.csv', columns=['id','label'], index=False)
