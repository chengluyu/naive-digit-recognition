import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

def normalize(dataset, area):
    return dataset.reshape(dataset.shape[0], area).astype('float32') / 255

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]

    x_train, x_test = normalize(x_train, num_pixels), normalize(x_test, num_pixels)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)

    print('Baseline error: %.2f%%' % (100 - scores[1] * 100))
