from keras.models import Sequential, load_model
from keras.layers import LSTM as LSTM_Layer, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np

from sklearn.metrics import accuracy_score


class LSTM:
    def __init__(self, id, num_classes, input_shape, dense_layers=None, lstm_layers=None, learning_rate=0.001):

        if dense_layers is None:
            dense_layers = [128]
        if lstm_layers is None:
            lstm_layers = [256]
        assert len(lstm_layers) > 0, "At least one LSTM layer is required."

        self.id = id
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dense_layers = dense_layers  # List of layer sizes
        self.lstm_layers = lstm_layers  # List of layer sizes

    def create_model(self):
        model = Sequential()
        i = 1
        model.add(LSTM_Layer(self.lstm_layers[0], input_shape=self.input_shape,
                             return_sequences=False if len(self.lstm_layers) == i else True))
        for lstm_layer in self.lstm_layers[1:]:
            i += 1
            model.add(LSTM_Layer(lstm_layer, return_sequences=False if len(self.lstm_layers) == i else True))

        for dense_layer in self.dense_layers:
            model.add(Dense(dense_layer, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, X_valid, y_valid, epochs=100, batch_size=64):
        chk = ModelCheckpoint('best_lstm_' + self.id + '.pkl', monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[chk], validation_data=(X_valid, y_valid))

    def evaluate(self, X, y):
        model = load_model('best_lstm_' + self.id + '.pkl')
        return accuracy_score([np.argmax(t) for t in y], model.predict_classes(X))
