import numpy as np
import zmq
import tensorflow as tf
import json
import pandas as pd
import itertools as it
from classes.datagen import TrialGenerator

class LoggerCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, validation_data, saver=None, port=5555,
                 predictions=pd.DataFrame(), history=pd.DataFrame(),
                 store_weights=False):
        self._model = model
        self._validation_data = validation_data
        self._formated_validation = TrialGenerator.format_validation(validation_data)
        self._predictions = predictions.values.tolist()
        self._history = [h for h in history.T.to_dict().values()]
        self._epocs = []

        if store_weights:
            self._weights = np.zeros((0, 128, 128))
        else:
            self._weights = None

        if port is not None:
            self._cues = validation_data['cues'].reindex(np.arange(4), level='trial')
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUSH)
            self._socket.connect(f"tcp://localhost:{port}")

            message = json.dumps(list(self._cues.astype(float))).encode()
            self._socket.send_multipart(['clear'.encode(), 'cues'.encode(), message])
        else:
            self._socket = None

        self._saver = saver
        self._i = 0

    @property
    def predictions(self):
        return self._predictions

    @property
    def weights(self):
        return self._weights

    def on_epoch_end(self, epoc, logs=None):
        self._predictions.append(
            self.model.predict(self._formated_validation[0]).squeeze())
        if self._weights is not None:
            self._weights = np.concatenate((self._weights, [self.model.layers[0].weights[1]]))

        if self._socket is not None:
            if epoc % 25 == 0:
                message = json.dumps(list(self._cues.astype(float))).encode()
                self._socket.send_multipart(['cues'.encode(), message])
            message = json.dumps(list(self.predictions[-1][:800].astype(float))).encode()
            self._socket.send_multipart(
                ['epoc'.encode(), str(self._i + 1).encode(),
                 'prediction'.encode(), message,
                 'val_mse'.encode(), str(logs['val_mean_squared_error']).encode()])

        self._epocs.append(epoc)

        if self._saver is not None:
            self._history.append(logs)

            if epoc %25 == 0:
                history = pd.DataFrame(self._history, index=self._epocs)
                self._saver.save(self.predictions, history, self._validation_data)
        self._i += 1


    def close_socket(self):
        if self._socket is not None:
            self._socket.close()
            self._context.term()


class UntrainedCB(tf.keras.callbacks.Callback):
    def __init__(self, saver, port = None):
        super(UntrainedCB, self).__init__()
        self._saver = saver
        self._port = port


    def on_train_begin(self, logs=None):
        self._saver.save()
        self.model.stop_training = True
