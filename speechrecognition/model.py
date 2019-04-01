from preprocess import *
import keras
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import InputLayer, Input
from keras.applications.inception_v3 import InceptionV3

from keras.utils import to_categorical
import utils.kiel_corpus_utils as kcu
from keras.models import load_model
from keras import applications

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from utils.sampa_utils import SampaMapping
import pickle
from sklearn.decomposition import PCA










np.set_printoptions(threshold=np.nan)




def make_model(num_classes):



    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(12, 12, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dens_1'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])



    model.save_weights("first_mod")



    return model

def make_transfer_model(weights_file,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(12, 12, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dens_1'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(weights_file, by_name=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])



    return model


def make_model_svm(num_classes):



    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(12, 12, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dens_1'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # model.save_weights('my_model_weights.h5')

    return model



def exp_model(num_classes):
    model = Sequential()

    model.add(Conv2D(64, (2, 2), activation='relu', name='block1_conv1', input_shape=(12, 12, 1)))
    model.add(Conv2D(64, (2, 2), activation='relu', name='block2_conv2768'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool8'))
    model.add(Dropout(0.25))



    model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv1'))
    model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(Dropout(0.25))

    model.add(Flatten())


    model.add(Dense(128, activation='relu', name='dens_1'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))


    model.save_weights('smallset.h5')
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])



    return model

def exp_model_trans(weights_file, num_classes):
    model = Sequential()

    model.add(Conv2D(64, (2, 2), activation='relu', name='block1_conv1', input_shape=(12, 12, 1), trainable=False))
    model.add(Conv2D(64, (2, 2), activation='relu', name='block2_c'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_peep'))
    model.add(Dropout(0.25))



    model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv1546'))
    model.add(Conv2D(128, (2, 2), activation='relu', name='block2_c2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_p'))
    model.add(Dropout(0.25))

    model.add(Flatten())


    model.add(Dense(128, activation='relu', name='den'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))



    model.load_weights(weights_file, by_name=True)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])



    return model




















def exp_model(num_classes):
   model = Sequential()

   model.add(Conv2D(64, (2, 2), activation='relu', name='block1_conv1', input_shape=(12, 12, 1)))
   model.add(Conv2D(64, (2, 2), activation='relu', name='block2_conv2768'))
   model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool8'))
   model.add(Dropout(0.25))



   model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv1'))
   model.add(Conv2D(128, (2, 2), activation='relu', name='block2_conv2'))
   model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
   model.add(Dropout(0.25))

   model.add(Flatten())


   model.add(Dense(128, activation='relu', name='dens_1'))
   model.add(Dropout(0.25))
   model.add(Dense(num_classes, activation='softmax'))

   model.save_weights('AnetworkA.h5')
   model.summary()

   model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adadelta(),
                 metrics=['accuracy'])



   return model


















    
# Predicts one sample
# demodel = Model(inputs=base_inception.input, outputs=predictions)f predict(filepath, model):
#     sample = wav2mfcc(filepath)
#   # only if we want to freeze layers  sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
#   for layer in base_inception.layers:  return get_labels()[0][
#       layer.trainable = False          np.argmax(model.predict(sample_reshaped))
#     ]




if __name__ == '__main__':

    #transfer_imgnet(67)
    dir_path = '/Users/samski/Downloads/kiel_corpus 2/kiel_corpus/hello'
    # feats, labs = spectrograms_features_and_labels_from_dir_kiel(dir_path, (3, 3))
    feats, labs = kcu.spectrograms_features_and_labels_from_dir_kiel(dir_path, (12, 12))

    # 
    # dir_path = '/Users/samski/Downloads/kiel_corpus_3'
    # # feats, labs = spectrograms_features_and_labels_from_dir_kiel(dir_path, (3, 3))
    # feats, labs = kcu.spectrograms_features_and_labels_from_dir_kiel(dir_path, (12, 12))
    X_train, X_test, y_train, y_test = get_train_test(feats, labs)
    #
    # print(feats.shape)
    # print(labs.shape)
    #
    feature_dim_1 = 20
    channel = 1
    epochs = 1
    batch_size = 64
    verbose = 1
    num_classes = 67


    #
    # X_train = np.load("xtrain3.npy")
    # X_test = np.load("xtest3.npy")
    # y_train = np.load("ytrain3.npy")
    # y_test = np.load("ytest3.npy")
    # # #
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_train_hot = to_categorical(y_train, num_classes)
    y_test_hot = to_categorical(y_test, num_classes)


    # model =  exp_model_trans('wholeset.h5', num_classes)

    model = exp_model(num_classes)
    # model.load_weights('wholeset2.h5')


    # history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_hot), verbose=1)
    # print(history.history.keys())
    # with open('NETB_thirdconv_UNTRAIN.pickle', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)
    # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()









    #CITE

    model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
    # #
    #
    svm_x_train = []
    svm_y_train = []
    for i in range(len(X_train)):
        x_1 = np.expand_dims(X_train[i], axis=0)
        # print(x_1.shape)
        flatten_2_features = model2.predict(x_1)# # model = make_model(num_classes)
        #
         # model.save_weights('hello.h5')                  # print(flatten_2_features.shape)
        svm_x_train.append(flatten_2_features)# #
        svm_y_train.append(y_train_hot[i])


    svm_x_train = np.array(svm_x_train)
    clf = svm.SVC(kernel='rbf', class_weight='balanced', C = 1000, gamma = .01)
    dataset_size = len(svm_x_train)
    svm_x_train = np.array(svm_x_train).reshape(dataset_size,-1)
    svm_y_train = np.array(svm_y_train)
    svm_y_train = [np.where(r==1)[0][0] for r in svm_y_train]

    clf.fit(svm_x_train, svm_y_train)



    # %time grid.fit(Xtrain, ytrain)
    # print(grid.best_params_)

    print(svm_x_train.shape)
    print(len(svm_y_train))

    svm_x_test = []
    svm_y_test = []
    for i in range(len(X_test)):
        x_1 = np.expand_dims(X_test[i], axis=0)
        #x_1 = preprocess_input(x_1)
        flatten_2_features = model2.predict(x_1)    # # model.fit(X_train, y_train_hot, batch_size = batch_size, epochs=epochs, validation_data=(X_test, y_test_hot))
        svm_x_test.append(flatten_2_features)# #
        svm_y_test.append(y_test_hot[i])# #

    svm_x_test = np.array(svm_x_test)

    dataset_size = len(svm_x_test)
    svm_x_test = np.array(svm_x_test).reshape(dataset_size,-1)
    svm_y_test = [np.where(r==1)[0][0] for r in svm_y_test]

    print(svm_x_test.shape)
    #
    # print(svm_y_test)

    yfit = clf.predict(svm_x_test)
    # print(yfit)
    yfit_hot = to_categorical(yfit, num_classes)
    print(yfit_hot.shape)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(svm_y_test, yfit))
    #
    # pca = PCA(2)  # project from 64 to 2 dimensions
    # projected = pca.fit_transform(yfit_hot)
    # print(yfit.shape)
    # print(projected.shape)
    #
    # plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5, c=yfit,
    #             cmap=plt.cm.get_cmap('nipy_spectral', 10))
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.colorbar();
    # plt.show()





    # from sklearn.svm import SVC
    # from sklearn.decomposition import RandomizedPCA
    # from sklearn.pipeline import make_pipeline
    #
    # pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
    # svc = SVC(kernel='rbf', class_weight='balanced')
    # model = make_pipeline(pca, svc)
    #
    # svm_y_train = []
    # svm_x_train = []
    #
    # for i in range(len(X_train)):
    #     svm_y_train.append(y_train_hot[i])  # #
    #     svm_x_train.append()
    #
    #
    # print(len(svm_y_train))
    #
    # svc.fit(X_train, svm_y_train)












    #
    #
    #
    #
    # print(np.array([SampaMapping.idx2sampa[s] for s in svm_y_test]))
    #
    # yfit = clf.predict(svm_x_test)
    #
    # print(np.array([SampaMapping.idx2sampa[s] for s in yfit]))
    # #
    #
    # from sklearn.metrics import accuracy_score
    # print(accuracy_score(svm_y_test, clf.predict(svm_x_test)))
    # #
    from sklearn.metrics import classification_report

    target_names = list(SampaMapping.idx2sampa.values())
    # print(target_names)

    print(classification_report(svm_y_test, yfit, target_names=target_names))

    # from sklearn.metrics import confusion_matrix
    #
    # mat = confusion_matrix(svm_y_test, yfit)
    # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    # plt.xlabel('true label')
    # plt.ylabel('predicted label');
    # plt.show()






    #
    #
    # #
    # # model = make_transfer_model("first_mod",num_classes)
    # #
    # # model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_hot))
    #
    # #TODO: Add dense layer, get summary, make all three transfer nets with weights, add in tensorboard!, how to graph accuracy of CNN's?
    #
    # model = exp_model(num_classes)
    #
    # model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_hot))

    # model = exp_model_trans("sec_mod", num_classes)
    # model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_hot))
    #
    #
    #







