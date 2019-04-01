#from https://github.com/manashmndl/DeadSimpleSpeechRecognizer/blob/master/preprocess.py

import numpy as np
from sklearn.model_selection import train_test_split
from utils.kiel_corpus_utils import spectrograms_features_and_labels_from_dir_kiel


def get_train_test(features, labels,split_ratio=0.6, random_state=42):

    return train_test_split(features, labels, test_size=(1 - split_ratio), random_state=random_state)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)

    dir_path = '/Users/samski/Downloads/hello'
    feats, labs = spectrograms_features_and_labels_from_dir_kiel(dir_path, (12, 12))

    print(feats.shape)
    print(labs.shape)

    # np.save("feats2.npy", feats)
    # np.save("labs2.npy", labs)
    #
    # feats = np.load("feats.npy")
    # labs = np.load("labs.npy")

    # print(feats.shape)
    # print(labs.shape)
    tt_split = get_train_test(feats, labs)

    X_train = tt_split[0]
    X_test  = tt_split[1]
    y_train = tt_split[2]
    y_test = tt_split[3]

    np.save("xtrainNETB.npy", X_train)
    np.save("xtestNETB.npy", X_test)
    np.save("ytrainNETB.npy", y_train)
    np.save("ytestNETB.npy", y_test)

