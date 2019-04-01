import matplotlib.pyplot as plt
import seaborn as sns
from utils.sampa_utils import SampaMapping
import pickle


if __name__ == '__main__':

    file1 = open("NETA.pickle", 'rb')
    history = pickle.load(file1)
    file1.close()

    file2 = open("NETB_thirdconv_UNTRAIN.pickle", 'rb')
    history2 = pickle.load(file2)
    file2.close()

    # file3 = open("NETB_firstconv.pickle", 'rb')
    # history3 = pickle.load(file3)
    # file3.close()
    #
    # file4 = open("NETB_thirdconv.pickle", 'rb')
    # history4 = pickle.load(file4)
    # file4.close()




    print(history.keys())

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.plot(history2['acc'], marker='o')
    plt.plot(history2['val_acc'], marker='o')
    #plt.plot(history3['acc'], marker='1')
    # plt.plot(history3['val_acc'], marker='1')
    # plt.plot(history4['acc'], marker='x')
    # plt.plot(history4['val_acc'], marker='x')
    plt.title('Accuracy for Original and Transfer Networks (frozen weights)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Net A train', 'Net A test', 'Net B train', 'Net B test'], loc='lower right')
    plt.show()