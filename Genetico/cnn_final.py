from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cPickle, gzip, time
import numpy as np

np.random.seed(1337)

cont = 0
file_results = open('results.txt', 'w')

def cnn(list):
    start_time = time.time()

    #####----- Parameters -----#####
    batch_size = 30
    nb_classes = 2
    nb_epoch = 30

    img_rows, img_cols = 28,28
    nb_filters_1 = list[0]
    nb_filters_2 = list[1]
    nb_in_neurons = (list[1] * 48)
    nb_hidden_neurons = list[2]
    nb_drop_cnn = 0.25
    nb_drop_mlp = 0.5
    nb_pool = 2
    nb_conv = 5
    global cont

    file_epochs = open('Epochs/epochs_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.txt', 'w')

    predicts = open('Predicts/predicts_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.txt', 'w')
    ################################

    #####----- Datasets -----#####
    train_set_original, test_set_original = cPickle.load(gzip.open('lung_nodule_Original_3.pkl.gz', 'rb'))
    (X_train_original, y_train_original), (X_test_original, y_test_original) = train_set_original, test_set_original

    X_train_original = X_train_original.reshape(X_train_original.shape[0], 1, img_rows, img_cols)
    X_test_original = X_test_original.reshape(X_test_original.shape[0], 1, img_rows, img_cols)
    X_train_original = X_train_original.astype('float32')
    X_test_original = X_test_original.astype('float32')
    X_train_original /= 255
    X_test_original /= 255

    Y_train_original = np_utils.to_categorical(y_train_original, nb_classes)
    Y_test_original = np_utils.to_categorical(y_test_original, nb_classes)

    train_set_external, test_set_external = cPickle.load(gzip.open('lung_nodule_Region_1_3.pkl.gz', 'rb'))
    (X_train_external, y_train_external), (X_test_external, y_test_external) = train_set_external, test_set_external

    X_train_external = X_train_external.reshape(X_train_external.shape[0], 1, img_rows, img_cols)
    X_test_external = X_test_external.reshape(X_test_external.shape[0], 1, img_rows, img_cols)
    X_train_external = X_train_external.astype('float32')
    X_test_external = X_test_external.astype('float32')
    X_train_external /= 255
    X_test_external /= 255

    train_set_internal, test_set_internal = cPickle.load(gzip.open('lung_nodule_Region_2_3.pkl.gz', 'rb'))
    (X_train_internal, y_train_internal), (X_test_internal, y_test_internal) = train_set_internal, test_set_internal

    X_train_internal = X_train_internal.reshape(X_train_internal.shape[0], 1, img_rows, img_cols)
    X_test_internal = X_test_internal.reshape(X_test_internal.shape[0], 1, img_rows, img_cols)
    X_train_internal = X_train_internal.astype('float32')
    X_test_internal = X_test_internal.astype('float32')
    X_train_internal /= 255
    X_test_internal /= 255

    ##############################

    #####----- Convolutional Neural Networks -----#####
    original = Sequential()

    original.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    original.add(Activation('relu'))
    original.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    original.add(Convolution2D(nb_filters_2, nb_conv, nb_conv))
    original.add(Activation('relu'))
    original.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    original.add(Dropout(nb_drop_cnn))

    external = Sequential()

    external.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    external.add(Activation('relu'))
    external.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    external.add(Convolution2D(nb_filters_2, nb_conv, nb_conv))
    external.add(Activation('relu'))
    external.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    external.add(Dropout(nb_drop_cnn))

    internal = Sequential()

    internal.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    internal.add(Activation('relu'))
    internal.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    internal.add(Convolution2D(nb_filters_2, nb_conv, nb_conv))
    internal.add(Activation('relu'))
    internal.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    internal.add(Dropout(nb_drop_cnn))
    ###################################################

    #####----- Multi-Layer Perceptron -----#####
    mlp = Sequential()

    mlp.add(Merge([original, external, internal], mode='concat', concat_axis=1))

    mlp.add(Flatten())
    mlp.add(Dense(nb_in_neurons))
    mlp.add(Activation('relu'))
    mlp.add(Dropout(nb_drop_mlp))
    mlp.add(Dense(nb_hidden_neurons))
    mlp.add(Activation('relu'))
    mlp.add(Dense(nb_classes))
    mlp.add(Activation('softmax'))

    mlp.compile(loss='binary_crossentropy', optimizer='sgd')

    train = mlp.fit([X_train_original, X_train_external, X_train_internal], Y_train_original, batch_size=batch_size,
                    nb_epoch=nb_epoch, show_accuracy=True, verbose=1,  validation_split=0.1)
    score = mlp.evaluate([X_test_original, X_test_external, X_test_internal], Y_test_original, batch_size=batch_size,  show_accuracy=True, verbose=0)
    ############################################

    file_epochs.write(str(train.history) + "\n\n")

    file_epochs.write("Test Loss and Test Accuracy: ")
    file_epochs.write(str(score) + "\n\n")

    listPredict = []

    listPredict = mlp.predict_classes([X_test_original, X_test_external, X_test_internal])

    vp, fp, vn, fn = 0, 0, 0, 0

    for i in xrange(X_test_original.shape[0]):
        predicts.write(str(listPredict[i]) + "\n")
        if y_test_original[i] == 0 and listPredict[i] == 0:
            vn += 1
        elif y_test_original[i] == 0 and listPredict[i] == 1:
            fp += 1
        elif y_test_original[i] == 1 and listPredict[i] == 1:
            vp += 1
        else:
            fn += 1

    sen = float((vp * 1.0)/(vp + fn)) #malignant

    esp = float((vn * 1.0)/(vn + fp)) #benign

    print score[1], sen, esp, nb_filters_1, nb_filters_2, nb_hidden_neurons

    ####--------- Saving Models ----------####

    original.save_weights('Model_Original/weights_o_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.h5')

    external.save_weights('Model_External/weights_e_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.h5')

    internal.save_weights('Model_Internal/weights_i_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.h5')

    mlp.save_weights('Model_MLP/weights_m_' + str(nb_filters_1) + '_' + str(nb_filters_2) + '_' + str(nb_hidden_neurons) +
                     '_' + str(cont) + '.h5')

    ##########################################

    file_epochs.write('Sen = ' + str(sen) + ' Esp = ' + str(esp))

    cont += 1

    file_epochs.close()

    predicts.close()

    fitness = (score[1] + esp + (sen * 2))

    print fitness

    file_results.write('Acc: ' + str(score[1]) + ' Sen: ' + str(sen) + ' Esp: ' + str(esp) +
                       ' Conv1: ' + str(nb_filters_1) + ' Conv2: ' + str(nb_filters_2) +
                       ' Hidden: ' + str(nb_hidden_neurons) + ' Fitness: ' + str(fitness) + ' Cont: ' + str(cont) + ' Time: ' + str(time.time() - start_time) +  '\n')

    return fitness