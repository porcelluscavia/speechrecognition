from preprocess import *
import keras
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import utils.kiel_corpus_utils as kcu

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn import svm
from sklearn import preprocessing

# model = Sequential()
# 
# model.add(Conv2D(64, (2, 2), activation='relu', name='block1_conv1', input_shape=(12, 12, 1)))
# model.add(Conv2D(64, (2, 2), activation='relu', name='block2_conv2768'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool8'))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv1'))
# model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# 
# model.add(Dense(128, activation='relu', name='dens_1'))
# model.add(Dropout(0.25))
# model.add(Dense(num_classes, activation='softmax'))
# 
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# 
# model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
# model.load_weights('my_model_weights.h5')
# 
# clf = svm.SVC(kernel='rbf', class_weight='balanced')
import subprocess
import matplotlib.pyplot
import os
import matplotlib.pyplot as plt


def hello():

    return True



# def train(x_train, y_train):
    #
    # encoder = preprocessing.LabelEncoder()
    # y_temp_train = y_train
    # encoder.fit(y_temp_train)
    # encoded_Y = encoder.transform(y_temp_train)
    # dummy_y = np_utils.to_categorical(encoded_Y)

    #
	# svm_x_train = []
	# svm_y_train = []
	# for i in range(len(x_train)):
	# 	x_1 = np.expand_dims(x_train[i], axis=0)
	# 	flatten_2_features = model2.predict(x_1)
	# 	svm_x_train.append(flatten_2_features)
	# 	svm_y_train.append(dummy_y[i])
    #
	# svm_x_train = np.array(svm_x_train)
	# clf = svm.SVC(kernel='rbf', class_weight='balanced')
	# dataset_size = len(svm_x_train)
	# svm_x_train = np.array(svm_x_train).reshape(dataset_size,-1)
	# svm_y_train = np.array(svm_y_train)
	# svm_y_train = [np.where(r==1)[0][0] for r in svm_y_train]
    #
    #
	# clf.fit(svm_x_train, svm_y_train)
	# print('model trained')
	# return clf
    #
    #

if __name__ == '__main__':
     # print('hello!')
     #
     # dir_path = '/Users/samski/Downloads/kiel_corpus_3/hello'
     #    # feats, labs = spectrograms_features_and_labels_from_dir_kiel(dir_path, (3, 3))
     # feats, labs = kcu.spectrograms_features_and_labels_from_dir_kiel(dir_path, (12, 12))
     #
     # print(feats.shape)
     # print(labs.shape)
     #    #
     #    # feature_dim_1 = 20
     #    # channel = 1
     #    # epochs = 50
     #    # batch_size = 64
     #    # verbose = 1
     #    # num_classes = 67
     #
     # X_train, X_test, y_train, y_test = get_train_test(feats, labs)
     #
     # print(X_train.shape)
     # print(y_train.shape)
     # print(X_test.shape)
     #    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
     #    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
     #    # y_train_hot = to_categorical(y_train, num_classes)
     #    # y_test_hot = to_categorical(y_test, num_classes)
     #
     #
     # encoder = preprocessing.LabelEncoder()
     # y_temp_train = y_train
     # print(y_train)
     #
     # print(encoder.fit(y_temp_train))
     # encoded_Y = encoder.transform(y_temp_train)
     # print(encoded_Y)
     #    # dummy_y = np_utils.to_categorical(encoded_Y)encoded_Y

     np.set_printoptions(threshold=np.nan)

     from sklearn.datasets import load_digits

     digits = load_digits()
     print(digits.target)