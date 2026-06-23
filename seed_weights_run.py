import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
import argparse

import itertools as it
from classes.models import LeakyRNNCell, LeakyRNN
from classes.callbacks import LoggerCallback, UntrainedCB
from classes.datagen import tDNMSGenerator, tDNMSGeneratorLL, SLOnlyTask, CueResponse, tDNMS_no_match, tDNMSRespAll, genFactory
from classes.utils import ModelSaver
import models_database as mdb


def load_eigs():
    no_model_ids   = [9,  18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79,  82,  84]
    example_ids = [18, 20]

    idx = pd.MultiIndex.from_tuples(
        list(zip(no_model_ids, it.repeat('no_shaping'))) +
        list(zip(with_model_ids, it.repeat('shaping'))),
        names=('model_id', 'model_type'))
    model_ids = pd.Series(no_model_ids + with_model_ids,  index=idx)

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_attrs = model_ids.apply(mdb.get_model_attributes)
    model_infos = pd.concat(
        (model_infos,
         model_attrs[['gamma', 'epoc', 'input_noise', 'noise_level']]), axis=1)

    models = model_infos['path'].apply(tf.keras.models.load_model)
    weights = pd.concat(models.apply(
        lambda x: x.layers[0].get_weights()[1]).apply(pd.DataFrame).values, keys=models.index)
    eigs = weights.groupby(['model_id', 'model_type']).apply(np.linalg.eig)
    eig_vect = eigs.apply(lambda x: x[1])
    eigs = eigs.apply(lambda x: x[0])
    eigs = eigs.apply(lambda x: np.sort(x)[::-1])
    return eigs.xs('shaping', level='model_type').apply(lambda x: x[:4]).apply(pd.Series).mean()

 
def main(argv):
    seed = load_eigs()
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', default='just_short_match', type=str,
                        help="string, name of the task to run, needs to match" +
                             " a task name in the genFactory class of" +
                             " classes/datagen.py and if first time task is" +
                             " run use '--create_task' to add to the database")
    parser.add_argument('--base_id', '-b', type=int, default=0,
                        help='model_id from the database to initalize weights' +
                             ' for training. if 0, generate a new model with' +
                             ' no prior training')
    parser.add_argument('--epochs', '-e', default=1000, type=int)
    parser.add_argument('--units', '-u', default=128, type=int)
    parser.add_argument('--create_task', action='store_true')
    parser.add_argument('--create_project', action='store_true')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_load_params', action='store_true')
    parser.add_argument('--store_weights', action='store_true')
    parser.add_argument('--no_update', '-X', action='store_true')
    args = parser.parse_args(argv)

    models_folder = '/models'
    callback_port = None
    store_weights = args.store_weights
    update_database = not args.no_update

    base_id = args.base_id
    task = args.task
    task_name = task + '_seed'
    units = args.units
    no_train = args.no_train
    project = 'rnn_shaping'
    if args.create_task:
        mdb.fetch_task_id(task_name, create=True)
    if args.create_project:
        mdb.fetch_project_id(project, create=True)
    max_epocs = args.epochs

    if base_id == 0 or args.no_load_params:
        params = {
            'activation': 'tanh',
            'weights_regularizer_coef': 1e-3,
            'activity_regularizer_coef': 1e-3,
            'learning_rate': 1e-5,
            'noise_level': 0.3,
            'input_noise': 0.15,
            'gamma': 0.2,
            'input_blocks_per_epoch': 1,
            'batch_size': 2,
        }
    else:
        params = mdb.get_model_attributes(base_id)
        keys = ['activation',
                'weights_regularizer_coef',
                'activity_regularizer_coef',
                'learning_rate',
                'noise_level',
                'input_noise',
                'gamma',
                'input_blocks_per_epoch',
                'batch_size'
                ]
        params = params[keys].to_dict()

    params['input_blocks_per_epoch'] = int(params['input_blocks_per_epoch'])
    if update_database:
        model_id = mdb.insert_model(project, task_name, models_folder,
                                    base_id=base_id, attrs=params)
    else:
        model_id = -1

    if base_id != 0:
        model_info = mdb.get_model(base_id)
        model = tf.keras.models.load_model(model_info['path'])
    else:
        model = tf.keras.models.Sequential([
            LeakyRNN(units, activation=params['activation'],
                     gamma=params['gamma'], noise_std=params['noise_level'],
                     return_sequences=True,
                     activity_regularizer=tf.keras.regularizers.L2(
                         params['activity_regularizer_coef']),
                     recurrent_regularizer=tf.keras.regularizers.L2(
                         params['weights_regularizer_coef']),
                     kernel_regularizer=tf.keras.regularizers.L2(
                         params['weights_regularizer_coef'])),
            tf.keras.layers.Dense(
                units=1,
                activity_regularizer=tf.keras.regularizers.L2(
                    params['activity_regularizer_coef']),
                kernel_regularizer=tf.keras.regularizers.L2(
                    params['weights_regularizer_coef']))
        ])
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=params['learning_rate']),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    traingen = genFactory.create(
        task, params['input_noise'], params['batch_size'],
        params['input_blocks_per_epoch'])
    validate_data = traingen.generate_trials(10)
    formated_validation = tDNMSGenerator.format_validation(validate_data)

    model.build(input_shape=(None, None, 2))
    all_weights = model.layers[0].get_weights()
    e, v = np.linalg.eig(all_weights[1])

    new_e = e.copy()
    new_e[:4] = seed.values
    w2 = np.real(v @ np.diag(new_e) @ np.linalg.inv(v))
    all_weights[1] = w2.astype(np.float32)
    model.layers[0].set_weights(all_weights)

    saver = ModelSaver(model, models_folder, model_id, params)
    mdb.update_model(model_id, path=saver._model_path)

    all_history = []
    if no_train == False:
        # for training
        cb_train = LoggerCallback(
            model, validate_data, saver=saver, port=callback_port,
            store_weights=store_weights)
        for i in range(3):
            history = model.fit(
                traingen, validation_data=formated_validation, epochs=100,
                callbacks=cb_train)
            all_history.append(history.history)

            all_weights = model.layers[0].get_weights()
            e, v = np.linalg.eig(all_weights[1])
            new_e = e.copy()
            new_e[:4] = seed.values
            w2 = np.real(v @ np.diag(new_e) @ np.linalg.inv(v))
            all_weights[1] = w2.astype(np.float32)
            model.layers[0].set_weights(all_weights)
        history = model.fit(
            traingen, validation_data=formated_validation, epochs=700,
            callbacks=cb_train)
        all_history.append(history.history)

        history = pd.concat(list(map(pd.DataFrame, all_history)))

        cb_train.close_socket()
        saver.save(pd.DataFrame(cb_train.predictions),
                   history, validate_data)
        if store_weights:
            saver.save(
                pd.DataFrame(cb_train.predictions), history,
                validate_data, weights=cb_train.weights)
        else:
            saver.save(
                pd.DataFrame(cb_train.predictions), history,
                validate_data)

    else:
        # for no training
        cb_notrain = UntrainedCB(saver=saver, port=callback_port)
        history = model.fit(
            traingen, validation_data=formated_validation, epochs=max_epocs,
            callbacks=cb_notrain)
        
    if update_database:
        mdb.update_model(model_id, path=saver._model_path)

    mse = history['val_mean_squared_error'].tail(1).item()
    num_epocs = len(history['val_mean_squared_error'])
    if update_database:
        mdb.update_model_attr(model_id, 'mse', mse)
        mdb.update_model_attr(model_id, 'epoc', num_epocs)
        mdb.update_model_attr(model_id, 'seed_run', True)
    print(f'[{model_id}]')

if __name__ == '__main__':
    main(sys.argv[1:])
