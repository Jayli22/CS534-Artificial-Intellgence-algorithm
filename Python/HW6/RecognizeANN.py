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
        # 创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.show()


def main():
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    historyplot = LossHistory()
    batch_size = 256
    nb_classes = 10
    nb_epoch = 20

    # preprocessing
    X = images.reshape(images.shape[0], 28 * 28)
    y = np_utils.to_categorical(labels, 10)

    # split set
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.4, train_size=0.6, random_state=None)
    # sssV=StratifiedShuffleSplit(n_splits=10,test_size=0.15,train_size=0.6,random_state=None)
    for train_index, rest_index in sss.split(images, labels):
        # print(train_index,rest_index)
        validationSize = int(np.size(rest_index) * 0.375)
        # testSize=np.size(rest_index)-validationSize
        validation_index = rest_index[0:validationSize]
        test_index = rest_index[validationSize:]
        # print(train_index,validation_index,test_index)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        x_val = X[validation_index]
        y_val = y[validation_index]
        # print(X_train,y_train)
        # print("TRAIN:", train_index, "TEST:", test_index)

    # print(images)
    # print(labels)
    # print(images.shape)
    # print(labels.shape)
    # print(X)
    # print(y)
    # print(X.shape)
    print(X_train.shape[0], 'train smples')
    print(x_val.shape[0], 'val samples')
    print(X_test.shape[0], 'test samples')

    # declare model
    model = Sequential()

    # first layer
    model.add(Dense(28, input_shape=(28 * 28,), kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-6, axis=1))

    #
    #
    #
    # Fill in Model Here
    model.add(Dense(28, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-6, axis=1))

    model.add(Dense(28, kernel_initializer='uniform'))
    model.add(Activation('tanh'))
    model.add(BatchNormalization(epsilon=1e-6, axis=1))
    #
    #
    model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
    model.add(Activation('softmax'))

    # Compile Model
    # sgd = optimizers.SGD(lr=0.01,decay = 1e-6, momentum = 0.9,nesterov=True)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train Model
    history = model.fit(X_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        callbacks=[historyplot])

    loss, accuracy = model.evaluate(X_test, y_test)
    X_pred = model.predict(X_test, verbose=0)
    X_pred = np.argmax(X_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(history.history)
    cm = confusion_matrix(y_test, X_pred)
    print(cm)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    print((model.summary()))
    historyplot.loss_plot('epoch')
    index = 1
    for i in range(0,X_pred.shape[0]):
        if not X_pred[i] == y_test[i] :
                index += 1
                plt.subplot(4,4,index)
                p = X_test[i].reshape((28,28))
                plt.imshow(p, cmap=plt.get_cmap('gray'))
                print("True:", y_test[i],"False:",X_pred[i])
        if index >= 12:
            break

    plt.show()
    model.save("ANN_train_model_model.hw6")

def decision_tree():
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    historyplot = LossHistory()
    batch_size = 256
    nb_classes = 10
    nb_epoch = 20
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
        print(X_feature)
        print("Create finished")
    feature_create()

    y = np_utils.to_categorical(labels, 10)

    # split set create features
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.4, train_size=0.6, random_state=None)

    for train_index, rest_index in sss.split(images, labels):
        # print(train_index,rest_index)
        validationSize = int(np.size(rest_index) * 0.375)
        # testSize=np.size(rest_index)-validationSize
        validation_index = rest_index[0:validationSize]
        test_index = rest_index[validationSize:]
        # print(train_index,validation_index,test_index)
        X_train = X_feature[train_index]
        y_train = y[train_index]
        X_test = X_feature[test_index]
        y_test = y[test_index]
        x_val = X_feature[validation_index]
        y_val = y[validation_index]
        # print(X_train,y_train)
        # print("TRAIN:", train_index, "TEST:", test_index)
    dtree = tree.DecisionTreeClassifier(max_depth=20, random_state=17)
    dtree.fit(X_train, y_train)
    tree_pred = dtree.predict(X_test)
    print("accuracy:",accuracy_score(y_test,tree_pred))
    tree_pred = np.argmax(tree_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # tree_params = {'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
    #                'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64]}
    # tree_grid = GridSearchCV(dtree, tree_params, cv=5, n_jobs=-1,
    #                          verbose=True)
    # tree_grid.fit(X_train, y_train)
    # print(tree_grid.best_params_, tree_grid.best_score_)
    cm = confusion_matrix(y_test, tree_pred)
    print(cm)
    from joblib import dump, load
    dump(dtree,'decisiontree_model.h5')

# main()
decision_tree()
# def load_data (path='images.npy'):
