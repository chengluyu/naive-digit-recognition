from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(9):
    plt.subplot(331 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

plt.show()
