from __future__ import print_function
import cPickle
import gzip
import numpy as np

np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

file_epochs = open('epochs.txt', 'w')
predicts = open('predict.txt', 'w')

file_epochs.write("Model's Log: \n\n")

batch_size = 30
nb_classes = 2
nb_epoch = 30

img_rows, img_cols = 28, 28
nb_filters1 = 80
nb_filters2 = 15
nb_in_neurons = 240
nb_hidden_neurons = 66
nb_pool = 2
nb_conv = 5

file_epochs.write("Epochs: " + str(nb_epoch) + "\nImage Size: " + str(img_cols) + "\nBatch Size: " + str(batch_size) + "\n")

train_set, test_set = cPickle.load(gzip.open('lung_nodule_Region_1_3.pkl.gz', 'rb'))
(X_train, y_train), (X_test, y_test) = train_set, test_set

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

file_epochs.write("Train Samples: " + str(X_train.shape[0]) + "\nTest Samples: " + str(X_test.shape[0]) + "\n\nTraining and Validation:\n")

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters1, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(nb_filters2, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(nb_in_neurons))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_hidden_neurons))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd')

train = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

file_epochs.write(str(train.history) + "\n\n")

print('Test score:', score[0])
print('Test accuracy:', score[1])

listPredict = []

listPredict = model.predict_classes(X_test)

vp, fp, vn, fn = 0, 0, 0, 0

for i in xrange(X_test.shape[0]):
    predicts.write(str(listPredict[i]) + "\n")
    if y_test[i] == 0 and listPredict[i] == 0:
        vn += 1
    elif y_test[i] == 0 and listPredict[i] == 1:
        fp += 1
    elif y_test[i] == 1 and listPredict[i] == 1:
        vp += 1
    else:
        fn += 1

sen = float((vp * 1.0) / (vp + fn))

esp = float((vn * 1.0) / (vn + fp))

print (sen, esp)

file_epochs.write("Test Loss and Test Accuracy: ")
file_epochs.write(str(score) + "\n\n")

file_epochs.close()
predicts.close()

# json_string = model.to_json()
# open('model_architecture.json', 'w').write(json_string)
#model.save_weights('cemar_weights.h5', overwrite=True)