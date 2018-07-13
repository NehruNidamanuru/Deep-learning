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
