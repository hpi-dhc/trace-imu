from keras.models import Sequential, load_model
from keras.applications import Xception
from keras.layers import LSTM, Dense, TimeDistributed, Flatten, Reshape
from keras import Input, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import keras

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


class ResnetLSTM:
    def __init__(self, id, num_classes, input_shape=(4, 1, 64, 64), learning_rate=0.00001):
        self.id = id
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self):
        keras.backend.set_image_data_format('channels_first')
        resnet = Xception(include_top=False, pooling='max')
        resnet.trainable = False
        for layer in resnet.layers:
            layer.trainable = False
        input_layer = Input(shape=self.input_shape)
        curr_layer = TimeDistributed(resnet)(input_layer)
        lstm_out = LSTM(512)(curr_layer)
        out = Dense(self.num_classes, activation='softmax')(lstm_out)
        model = Model(inputs=input_layer, outputs=out)
        
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        self.model = model

    def fit(self, X, y, X_valid, y_valid, epochs=10, batch_size=32):
        chk = ModelCheckpoint('best_resnetlstm_' + self.id + '.pkl', monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1)
        '''
        print("Augmenting Data...")
        from tqdm import tqdm
        from scipy import ndimage
        import cv2
        X = list(X)
        y = list(y)
        n = len(X) * 3 // 2
        for i in tqdm(range(n)):
            flip = np.random.randint(0, 2)
            rot = np.random.randint(-5, 6)

            X.append([[ndimage.rotate(cv2.flip(img, flip), rot, reshape=False) for img in imgs] for imgs in X[i]])
            y.append(y[i])
        '''

        self.model.fit(np.array(X), np.array(y), epochs=epochs, batch_size=batch_size, callbacks=[chk],
                       validation_data=(X_valid, y_valid))

    def evaluate(self, X, y, enc):
        model = load_model('best_resnetlstm_' + self.id + '.pkl')
        y_pred = model.predict_classes(X)
        y_true = [np.argmax(t) for t in y]
        acc = accuracy_score(y_true, y_pred)
        conf = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
        #con_mat_norm = np.around(conf.astype('float') / conf.sum(axis=1)[:, np.newaxis], decimals=2)
        #con_mat_df = pd.DataFrame(con_mat_norm, index=enc, columns=enc)
        return acc, conf
