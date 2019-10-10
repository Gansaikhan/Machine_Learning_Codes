# MNIST is a classic 'Hello World' example in Machine Learning
# Written by Gansaikhan Shur
# (the original dataset can be found here http://yann.lecun.com/exdb/mnist/)

import os
from tensorflow import keras
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnistDataset = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnistDataset.load_data()

#print('Training data shape : ', x_train.shape, y_train.shape)
#print('Testing data shape : ', x_test.shape, y_test.shape)
"""
('Training data shape : ', (60000, 28, 28), (60000,))
('Testing data shape : ', (10000, 28, 28), (10000,))
"""
# To Visualize the dataset

# for i in range(9):
#    plt.subplot(3, 3, i+1)
#    plt.tight_layout()
#    plt.imshow(x_train[i], cmap='gray', interpolation='none')
#    plt.title("Digit: {}".format(y_train[i]))
#    plt.xticks([])
#    plt.yticks([])
# plt.show()

# Shape of the dataset
#('x_train shape', (60000, 28, 28))
#('y_train shape', (60000,))
#('x_test shape', (10000, 28, 28))
#('y_test shape', (10000,))

#! Normalizing the Input Data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# One-Hot Encoding
n_classes = 10
#print("No one-hot encoding: ", y_train.shape)
# Shape before one-hot encoding:  (60000,)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
#print("one-hot encoding: ", Y_train.shape)
# Shape after one-hot encoding:  (60000, 10)

mnist_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='linear',
           padding='same', input_shape=(28, 28, 1)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='linear'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

mnist_model.compile(loss='categorical_crossentropy',
                    metrics=['accuracy'], optimizer='adam')

# training the model and saving metrics in history
mnist_model.fit(x_train, Y_train,
                batch_size=128, epochs=20,
                verbose=2,
                validation_data=(x_test, Y_test))

# Save the model
save_dir = "/results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
mnist_model.save(model_path)
print('Saved trained model at %s ' % model_path)

mnist_model = load_model('keras_mnist.h5')
loss_and_metrics = mnist_model.evaluate(x_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

# create predictions
predicted_classes = mnist_model.predict_classes(x_test)

# correct and incorrect predictions
corr_ind = np.nonzero(predicted_classes == y_test)[0]
incorr_ind = np.nonzero(predicted_classes != y_test)[0]
print(len(corr_ind), " classified correctly")
print(len(incorr_ind), " classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7, 14)

# plot correct predictions
for i, correct in enumerate(corr_ind[:9]):
    plt.subplot(6, 3, i+1)
    plt.imshow(x_test[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Predicted: {}, Label: {}".format(predicted_classes[correct],
                                          y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorr_ind[:9]):
    plt.subplot(6, 3, i+10)
    plt.imshow(x_test[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Predicted {}, Label: {}".format(predicted_classes[incorrect],
                                         y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.show()
