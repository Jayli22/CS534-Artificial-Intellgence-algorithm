import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils
from keras.layers import BatchNormalization, Flatten

import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop
from keras import callbacks
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # create an image
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plot
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # set grid
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def decision_tree():
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    # historyplot = LossHistory()
    # batch_size = 256
    # nb_classes = 10
    # nb_epoch = 20
    # preprocessing
    X = images.reshape(images.shape[0], 28 * 28)
    X_feature = np.zeros([images.shape[0], 6])
    def feature_create():
        print("Creating features...")
        for i in range(0, images.shape[0]):
            leftindex = 27
            rightindex = 0
            topindex = 27
            bottomindex = 0
            sum = 0
            left = 0
            right = 0
            top = 0
            bottom = 0
            for j in range(0, 28):
                for k in range(0,28):
                    p = images[i][j][k]
                    # average
                    sum += images[i, j, k]
                    # average=sum/784

                    # left right
                    if k < 14:
                        left += (images[i, j, k] + 1)
                    else:
                        right += (images[i, j, k] + 1)
                    # lr=left/right


                    # top bottom
                    if j < 14:
                        top += (images[i, j, k] + 1)
                    else:
                        bottom += (images[i, j, k] + 1)
                    # tb=top/bottom
                    if p > 0:
                        if k < leftindex:
                            leftindex = k
                        if k > rightindex:
                            rightindex = k
                        if j < topindex:
                            topindex = j
                        if j > bottomindex:
                            bottomindex = j
                        X_feature[i][1] += 1
            X_feature[i, 0] = sum / 784
            X_feature[i, 4] = left / right
            X_feature[i, 5] = top / bottom
            X_feature[i][2] = rightindex - leftindex
            X_feature[i][3] = bottomindex - topindex
        # print(X_feature)
        print("Create finished")
    # feature_create()

    y = np_utils.to_categorical(labels, 10)

    # split set
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.75, random_state=None)
    sssV = StratifiedShuffleSplit(n_splits=10, test_size=0.15, train_size=0.6, random_state=None)
    for tv_index, test_index in sss.split(images, labels):
        X_test = X[test_index]
        y_test = y[test_index]
        X_tv = X[tv_index]
        y_tv = y[tv_index]

    for train_index, validation_index in sssV.split(X_tv, y_tv):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[validation_index]
        y_val = y[validation_index]

    # dtree = tree.DecisionTreeClassifier(max_depth=5
    dtree = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                           max_features=None, max_leaf_nodes=None, min_samples_leaf=3,
                           min_samples_split=5,
                           presort=False, random_state=3, splitter='best')
    dtree.fit(X_train, y_train)
    tree_pred = dtree.predict(X_test)
    print("accuracy:",accuracy_score(y_test,tree_pred))
    tree_pred = np.argmax(tree_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_val = np.argmax(y_val,axis=1)
    # tree_params = {'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
    #                'min_samples_split':[2,3,4,5,10,15,20,30,35],
    #                'min_samples_leaf':[2,3,4,5,6,7,8,15,20,30]}
    #                # 'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64]}
    # tree_grid = GridSearchCV(dtree, tree_params, cv=5, n_jobs=4,
    #                          verbose=True)
    # tree_grid.fit(X_train, y_train)
    # print(tree_grid.best_params_, tree_grid.best_score_)
    cm = confusion_matrix(y_test, tree_pred)
    print(cm)
    from joblib import dump, load
    dump(dtree, 'decisiontree_model.h5')

decision_tree()
# def load_data (path='images.npy'):
