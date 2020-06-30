from keras.models import Sequential, load_model
from keras.layers import ConvLSTM2D, MaxPooling3D, Dense, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


class ConvLSTM:
    def __init__(self, id, num_classes, input_shape=(4, 1, 64, 64),
                 dense_layers=None, conv_layers=None, augmentation=None,
                 learning_rate=0.00001):

        if dense_layers is None:
            dense_layers = [512, 256]
        if conv_layers is None:
            conv_layers = [(16, 5), (32, 3)]
        assert len(conv_layers) > 0, "At least one convolutional layer is required."

        self.id = id
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_layers = dense_layers  # List of layer sizes
        self.conv_layers = conv_layers  # List of tuples with (channels, kernel_size)
        self.augmentation = augmentation

    def create_model(self):
        model = Sequential()
        i = 1
        model.add(ConvLSTM2D(self.conv_layers[0][0], self.conv_layers[0][1],
                             data_format='channels_first', input_shape=self.input_shape,
                             return_sequences=False if len(self.conv_layers) == i else True))
        model.add(MaxPooling3D((1, 2, 2)))
        for conv_layer in self.conv_layers[1:]:
            model.add(ConvLSTM2D(conv_layer[0], conv_layer[1], data_format='channels_first',
                                 return_sequences=False if len(self.conv_layers) == i else True))
            model.add(MaxPooling3D((1, 2, 2)))
        model.add(TimeDistributed(Flatten()))
        for dense_layer in self.dense_layers:
            model.add(TimeDistributed(Dense(dense_layer, activation='relu')))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        self.model = model
        model.save('convlstm.h5')

    def fit(self, X, y, X_valid, y_valid, epochs=100, batch_size=32):
        chk = ModelCheckpoint('best_convlstm_' + self.id + '.pkl', monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1)
        if self.augmentation != {}:
            print("Augmenting Data...")
            from tqdm import tqdm
            from scipy import ndimage
            import cv2
            X = list(X)
            y = list(y)
            n = len(X) * 3 // 2
            for i in tqdm(range(n)):
                flipcodes = []
                if self.augmentation.get('horizontal_flip', False):
                    flipcodes.append(0)
                elif self.augmentation.get('vertical_flip', False):
                    flipcodes.append(1)
                if self.augmentation.get('horizontal_flip', False) and self.augmentation.get('vertical_flip', False):
                    flipcodes.append(-1)
                flip = np.random.choice(flipcodes)
                rot = np.random.randint(-self.augmentation.get('rotation_range',0), self.augmentation.get('rotation_range', 0)+1)

                X.append([[ndimage.rotate(cv2.flip(img, flip), rot, reshape=False) for img in imgs] for imgs in X[i]])
                y.append(y[i])
            

        self.model.fit(np.array(X), np.array(y), epochs=epochs, batch_size=batch_size, callbacks=[chk],
                       validation_data=(X_valid, y_valid))

    def evaluate(self, X, y, enc):
        model = load_model('best_convlstm_' + self.id + '.pkl')
        y_pred = model.predict_classes(X)
        y_true = [np.argmax(t) for t in y]
        acc = accuracy_score(y_true, y_pred)
        conf = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
        #con_mat_norm = np.around(conf.astype('float') / conf.sum(axis=1)[:, np.newaxis], decimals=2)
        #con_mat_df = pd.DataFrame(con_mat_norm, index=enc, columns=enc)
        return acc, conf
