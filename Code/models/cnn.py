from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from sklearn.metrics import accuracy_score


class CNN:
    def __init__(self, id, num_classes, input_shape=(1, 64, 64),
                 dense_layers=None, conv_layers=None, augmentation=None,
                 learning_rate=0.0001):

        if dense_layers is None:
            dense_layers = [256]
        if conv_layers is None:
            conv_layers = [(16, 5), (8, 3)]
        assert len(conv_layers) > 0, "At least one convolutional layer is required."

        self.id = id
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_layers = dense_layers  # List of layer sizes
        self.conv_layers = conv_layers  # List of tuples with (channels, kernel_size)
        self.augmentation = augmentation  # Dictionary, e.g. {'horizontal_flip': True, 'rotation_range': 90}

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(self.conv_layers[0][0], self.conv_layers[0][1], data_format='channels_first',
                         activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D())
        for conv_layer in self.conv_layers[1:]:
            model.add(Conv2D(conv_layer[0], conv_layer[1], data_format='channels_first', activation='relu'))
            model.add(MaxPooling2D())
        model.add(Flatten())
        for dense_layer in self.dense_layers:
            model.add(Dense(dense_layer, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, X_valid, y_valid, epochs=250, batch_size=32):
        chk = ModelCheckpoint('best_cnn_' + self.id + '.pkl', monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1)

        if self.augmentation == {}:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[chk],
                           validation_data=(X_valid, y_valid))
        else:
            datagen = ImageDataGenerator(**self.augmentation, fill_mode="nearest")
            datagen.fit(X)
            self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                                     epochs=epochs, callbacks=[chk], steps_per_epoch=len(X) // batch_size * 2,
                                     validation_data=(X_valid, y_valid))

    def evaluate(self, X, y):
        model = load_model('best_cnn_' + self.id + '.pkl')
        return accuracy_score([np.argmax(t) for t in y], model.predict_classes(X))
