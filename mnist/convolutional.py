import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('CNN Error: %.2f%%' % (100 - scores[1] * 100))
