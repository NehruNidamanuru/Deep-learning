import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


text=(open("/home/atchyuta/Problems/Smarttext/small.txt").read())
text=text.lower()
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

X = []
 Y = []
length = len(text)
seq_length = 100
  for i in range(0, length-seq_length, 1):
     sequence = text[i:i + seq_length]
     label =text[i + seq_length]
     X.append([char_to_n[char] for char in sequence])
     Y.append(char_to_n[label])


X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

history = model.fit(X_modified, Y_modified, validation_split=0.05, 
                    batch_size=batch_size, epochs=1, shuffle=True)

string_mapped = X[99]
# generating characters
for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]


This time we trained our model for 100 epochs and a batch size of 50
another LSTM layer with 400 units followed by a dropout layer of 0.2 fraction and see what we get.
I increased the number of units to 700 on each of the two LSTM layers.
I increased the number of layers to three, each having 700 units and trained it for 100 epochs
