import os

import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

class ModelSaver:

    def __init__(self, model, path, model_id, params={}):
        self._model = model
        base_path = os.path.join(path, f'model.{model_id}')

        if not (os.path.exists(base_path)):
            os.makedirs(base_path)

        self._model_path = os.path.join(base_path, f'model.{model_id}.keras')
        self._info_path = os.path.join(base_path, f'model.{model_id}.keras.dat.h5')
        self._weights_path = os.path.join(base_path, f'model.{model_id}.keras.weights.npz')
        self._txt_path = os.path.join(base_path, f'model.{model_id}.keras.info.txt')
        self._params=params

    def save(self, predictions=None, history=None, validation_data=None, weights=None):
        # save predictions if available
        if predictions is not None:
            pd.DataFrame(predictions).to_hdf(self._info_path, key='predictions')
        
        # save history if available
        if history is not None:
            history.to_hdf(self._info_path, key='history')
        
        # save validation data if available
        if validation_data is not None:
            validation_data.to_hdf(self._info_path, key='validation')
        
        # save the model architecture and weights
        self._model.save(self._model_path)

        # save parameters
        params = self._params.copy()
        
        if predictions is not None:
            params['epoc'] = len(predictions)
        
        if history is not None and 'val_mean_squared_error' in history:
            params['mse'] = history['val_mean_squared_error'].iloc[-1]

        # save the parameters to a text file
        params.update(self._params)
        with open(self._txt_path, 'w') as f:
            yaml.dump(params, f)

        if weights is not None:
            np.savez(self._weights_path, w=weights)

    @classmethod
    def load(cls, path, model_id):
        info_path = f'{path}.dat.h5'
        history = pd.read_hdf(info_path, 'history')
        predictions = pd.read_hdf(info_path, 'predictions')
        validation_data = pd.read_hdf(info_path, 'validation')
        return history, predictions, validation_data
