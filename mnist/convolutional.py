import numpy
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend
import cv2

def transform_nine(x_train, y_train):
    x_superset = []
    y_superset = []
    for x, y in zip(x_train, y_train):
        if y == 9 and random.random() < 0.5:
            upper = x[0:14,:]
            lower = x[14:,:]
            stretched = numpy.zeros(shape=(35, 28))
            stretched[0:14,:] = upper
            stretched[14:,:] = cv2.resize(lower, (28, 21), interpolation=cv2.INTER_AREA)
            stretched = cv2.erode(stretched, (7, 7))
            stretched = cv2.resize(stretched, (28, 28), interpolation=cv2.INTER_AREA)
            # cv2.imshow('debug', stretched)
            # cv2.waitKey()
            x_superset.append(stretched)
            y_superset.append(9)
        else:
            x_superset.append(x)
            y_superset.append(y)
    return numpy.array(x_superset), numpy.array(y_superset)

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = transform_nine(x_train, y_train)
    print('train x size: ' + str(x_train.shape[0]))
    print('train y size: ' + str(y_train.shape[0]))
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    backend.set_image_dim_ordering('th')
    (x_train, y_train), (x_test, y_test) = load_data()
    num_classes = y_test.shape[1]

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100, verbose=2)
    model.save('model.h5')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('CNN Error: %.2f%%' % (100 - scores[1] * 100))
