from keras.models import Sequential, load_model
from keras.layers import LSTM as LSTM_Layer, Conv1D, MaxPooling1D, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score


class ConvLSTM1D:
    def __init__(self, id, num_classes, input_shape, conv_layers=None, lstm_layers=None, learning_rate=0.001):

        if lstm_layers is None:
            lstm_layers = [512]
        if conv_layers is None:
            conv_layers = [(16, 5), (32, 3)]
        assert len(lstm_layers) > 0, "At least one LSTM layer is required."

        self.id = id
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_layers = conv_layers  # List of layer sizes
        self.lstm_layers = lstm_layers  # List of layer sizes

    def create_model(self):
        model = Sequential()
        i = 1
        model.add(Conv1D(self.conv_layers[0][0], self.conv_layers[0][1], activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(2))
        for conv_layer in self.conv_layers[1:]:
            model.add(Conv1D(conv_layer[0], conv_layer[1], activation='relu'))
            model.add(MaxPooling1D(2))
        model.add(LSTM_Layer(self.lstm_layers[0], input_shape=self.input_shape,
                             return_sequences=False if len(self.lstm_layers) == i else True))
        for lstm_layer in self.lstm_layers[1:]:
            i += 1
            model.add(LSTM_Layer(lstm_layer, return_sequences=False if len(self.lstm_layers) == i else True))

        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, X_valid, y_valid, epochs=100, batch_size=64):
        chk = ModelCheckpoint('best_convlstm1d_' + self.id + '.pkl', monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[chk], validation_data=(X_valid, y_valid))

    def evaluate(self, X, y, enc):
        model = load_model('best_convlstm1d_' + self.id + '.pkl')
        y_pred = model.predict_classes(X)
        y_true = [np.argmax(t) for t in y]
        acc = accuracy_score(y_true, y_pred)
        conf = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
        #con_mat_norm = np.around(conf.astype('float') / conf.sum(axis=1)[:, np.newaxis], decimals=2)
        #con_mat_df = pd.DataFrame(con_mat_norm, index=enc, columns=enc)
        return acc, conf
