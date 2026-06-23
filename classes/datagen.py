import tensorflow as tf
import pandas as pd
import numpy as np
import itertools as it
import random
import re

class TrialGenerator(tf.keras.utils.PyDataset):
    def __init__(self, cue_dict, target_dict, trial_length,
                 trial_blocks=None, batch_size=2,
                 input_noise=0.1, n_blocks=1, n_cues=1):
        ts_time = 3

        self._n_blocks = n_blocks
        if n_cues > 1:
            self._trial_cues = pd.concat(
                [pd.Series(list(b.T), name=a) for a, b in cue_dict.items()],
                           axis=1).stack().apply(pd.Series
                ).unstack(1).reorder_levels((1, 0), axis=1).sort_index(axis=1)
            self._trial_cues = self._trial_cues.reindex(np.arange(trial_length)).fillna(0).T
            self._trial_cues = self._trial_cues.unstack(1)
        else:
            self._trial_cues = pd.concat(
                [pd.Series(b, name=a) for a, b in cue_dict.items()], axis=1
                ).reindex(np.arange(trial_length)).fillna(0).T

        self._response_window = pd.concat(
            [pd.Series(b, name=a) for a, b in target_dict.items()], axis=1
            ).reindex(np.arange(trial_length)).reindex(
                self._trial_cues.index.unique(0), axis=1).fillna(0).T

        if trial_blocks == None:
            trial_blocks = list(self._trial_cues.index.get_level_values(0))
        trial_blocks = trial_blocks * n_blocks

        self._trial_start = \
            [1] * ts_time + [0]*(trial_length - ts_time)

        self._input_noise = input_noise
        self._window_length = trial_length
        self._trial_types = trial_blocks
        self._batch_size = batch_size
        self._n_cues = n_cues

        samples = len(self._response_window.reindex(trial_blocks).stack())
        self._length = (samples - trial_length) // batch_size
        self.prep_data()
        self._count = []

    @classmethod
    def format_validation(cls, data):
        odors = list(filter(re.compile('odor.*').search, data.columns))
        return  list(map(
            np.expand_dims,
            (data[['light'] + odors], data[['response']]),
            it.repeat(0)))

    @property
    def trial_cues(self):
        return self._trial_cues

    @property
    def response_window(self):
        return self._response_window

    def trial_starts(self, n):
        return pd.concat([self._trial_start] * n).reset_index(drop=True)

    def generate_trials(self, n, ts=0.1):
        if ts != 0.1:
            raise Exception('time steps not implemented')

        trial_list = pd.DataFrame(np.tile(self._trial_types, (n, 1))).apply(
            list, axis=1).apply(np.random.permutation).apply(
                pd.Series).stack().reset_index(drop=True)

        trials = trial_list.apply(lambda x: self.trial_cues.loc[x])
        trials.index = pd.MultiIndex.from_frame(trial_list.reset_index())
        trials = trials.stack(level=0, future_stack=True)
        trials.index.names = ['trial', 'type', 'idx']
        if len(trials.shape) == 1:
            X = trials.to_frame(name='cues')
            X['odor'] = X['cues'] + \
                        tf.random.normal((len(X['cues']),), mean=0,
                                         stddev=self._input_noise).numpy()
            cues = ['cues']
            odors = ['odor']
        else:
            cues = [f'cue_{a}' for a in trials.columns]
            trials.columns = cues
            X = trials
            for i, cue in enumerate(cues):
                X[f'odor_{i}'] = X[cue] + \
                                   tf.random.normal((X.shape[0],), mean=0,
                                                    stddev=self._input_noise).numpy()
            odors = [f'odor_{i}' for i, _ in enumerate(cues)]

        X['trial_start'] = np.tile(
            self._trial_start, (n * len(self._trial_types),))
        X['light'] = X['trial_start'] + \
                            tf.random.normal((X.shape[0],), mean=0,
                                             stddev=self._input_noise).numpy()
        X['response'] = trial_list.apply(
            lambda x: self.response_window.loc[x]).stack().values
        X = X.reindex(['trial_start', 'light'] + cues + odors + ['response'], axis=1)
        return X

    def data_splitter(self, features):
        inputs = features[:, slice(0, self._window_length), 0:(1 + self._n_cues)]
        labels = features[:, slice(0, self._window_length), (1 + self._n_cues):(2 + self._n_cues)]
        inputs.set_shape([None, self._window_length, None])
        labels.set_shape([None, self._window_length, None])
        return inputs, labels

    def prep_data(self):
        if self._n_cues > 1:
            data = np.array(self.generate_trials(1)[
                ['light'] + [f'odor_{a}' for a in range(self._n_cues)] + ['response']], dtype=np.float32)
        else:
            data = np.array(self.generate_trials(1)[
                ['light', 'odor', 'response']], dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self._window_length,
            sequence_stride=1,
            shuffle=False,
            batch_size=self._batch_size)

        ds = ds.map(self.data_splitter)
        self._ds_iter = iter(ds)
        self._ds = [d for d in ds]

    def on_epoch_end(self):
        self.prep_data()

    def __getitem__(self, index):
        return self._ds[index]

    def __len__(self):
        return self._length


class DNMS2CueGenerator(TrialGenerator):
    def __init__(self, trial_block=('M', 'NM1', 'NM2', 'M'), **kwargs):
        c1 = np.array([0] * 30 + [1] * 20 + [0] * 50)
        c2 = np.array([0] * 80 + [1] * 20)
        no_cue = [0] * 100
        cues = {'NM1': np.array([c1, c2]), 'NM2': np.array([c2, c1]), 'M': np.array([c1 + c2, no_cue])}

        response = np.zeros(150)
        response[101:120] = 1
        targets = {'NM1': response, 'NM2': response}
        super(DNMS2CueGenerator, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, n_cues=2, **kwargs)


class DNMS2CueGenerator_3input(TrialGenerator):
    def __init__(self, trial_block=('M', 'NM1', 'NM2', 'M'), **kwargs):
        c1 = np.array([0] * 30 + [1] * 20 + [0] * 50)
        c2 = np.array([0] * 80 + [1] * 20)
        no_cue = [0] * 100
        cues = {'NM1': np.array([no_cue, c1, c2]), 'NM2': np.array([no_cue, c2, c1]), 'M': np.array([no_cue, c1 + c2, no_cue])}

        response = np.zeros(150)
        response[101:120] = 1
        targets = {'NM1': response, 'NM2': response}
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, n_cues=2, **kwargs)


class tDNMSGenerator_3input(TrialGenerator):
    def __init__(self, trial_block=('SS', 'SL', 'LS', 'SS'), **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20 + [0] * 30
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        no_cue = [0] * 130
        cues = {
            'SS': np.array([ss, no_cue, no_cue]),
            'SL': np.array([sl, no_cue, no_cue]),
            'LS': np.array([ls, no_cue, no_cue])}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(tDNMSGenerator_3input, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, n_cues=3, **kwargs)


class tDNMSGenerator(TrialGenerator):
    def __init__(self, trial_block=('SS', 'SL', 'LS', 'SS'), **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(tDNMSGenerator, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, **kwargs)


class tDNMSTemporalJitter(TrialGenerator):
    def __init__(self, trial_block=('SS', 'SL', 'LS', 'SS'), jitter=0, **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}
        self.jitter = jitter * 10

        self._cue_dict = cues
        self._trial_length = 200

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(tDNMSTemporalJitter, self).__init__(
            cues, targets, self._trial_length, trial_blocks=trial_block, **kwargs)

    @property
    def trial_cues(self):
        c1_start_jitter = random.random() * (self.jitter * 2) - self.jitter / 2
        c1_stop_jitter = random.random() * (self.jitter * 2) - self.jitter / 2
        c2_start_jitter = random.random() * (self.jitter * 2) - self.jitter / 2
        c2_stop_jitter = random.random() * (self.jitter * 2) - self.jitter / 2

        ss = [0] * int(30 + c1_start_jitter) + \
             [1] * max(1, int(20 - c1_start_jitter + c1_stop_jitter)) + \
             [0] * int(30 - c1_stop_jitter + c2_start_jitter) + \
             [1] * max(1, int(20 - c2_start_jitter + c2_stop_jitter))
        sl = [0] * int(30 + c1_start_jitter) + \
             [1] * max(1, int(20 - c1_start_jitter + c1_stop_jitter)) + \
             [0] * int(30 - c1_stop_jitter + c2_start_jitter) + \
             [1] * max(1, int(50 - c2_start_jitter + c2_stop_jitter))
        ls = [0] * int(30 + c1_start_jitter) + \
             [1] * max(1, int(50 - c1_start_jitter + c1_stop_jitter)) + \
             [0] * int(30 - c1_stop_jitter + c2_start_jitter) + \
             [1] * max(1, int(20 - c2_start_jitter + c2_stop_jitter))
        cue_dict = {'SS': ss, 'SL': sl, 'LS': ls}

        trial_length = self._trial_length
        self._trial_cues = pd.concat(
            [pd.Series(b, name=a) for a, b in cue_dict.items()], axis=1
            ).reindex(np.arange(trial_length)).fillna(0).T

        return self._trial_cues


class SuperLongTrial(TrialGenerator):
    def __init__(self,  **kwargs):
        cue = [0] * 30 + [1] * 100 + [0] * 20 
        cues = {'super_long': cue}

        trial_block=('super_long', 'super_long', 'super_long', 'super_long')
        response = np.zeros(150)
        targets = {'super_long': response}
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, **kwargs)

class tDNMSGeneratorMM(TrialGenerator):
    def __init__(self, **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        mm = [0] * 30 + [1] * 35 + [0] * 30 + [1] * 35
        cues = {'SS': ss, 'SL': sl, 'LS': ls, 'MM': mm}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        trial_blocks = ('MM', 'SL', 'LS', 'SS')

        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_blocks, **kwargs)

class OmitOne(TrialGenerator):
    def __init__(self, **kwargs):
        ss = [0] * 30 + [0] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [0] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [0] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        trial_blocks = ('SS', 'SL', 'LS', 'SS')

        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_blocks, **kwargs)


class ProbeTaskGenerator(TrialGenerator):
    def __init__(self, trial_block=('SS', 'SL', 'LS', 'SS'), **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(tDNMSGenerator, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, **kwargs)


    def generate_trials(self, n, ts=0.1):
        if ts != 0.1:
            raise Exception('time steps not implemented')

        trial_list = pd.DataFrame(np.tile(self._trial_types, (n, 1))).apply(
            list, axis=1).apply(np.random.permutation).apply(
                pd.Series).stack().reset_index(drop=True)

        trials = trial_list.apply(lambda x: self.trial_cues.loc[x])
        trials.index = pd.MultiIndex.from_frame(trial_list.reset_index())
        trials = trials.stack()
        trials.index.names = ['trial', 'type', 'idx']
        X = trials.to_frame(name='cues')
        X['odor'] = X['cues'] + \
                    tf.random.normal((len(X['cues']),), mean=0,
                                     stddev=self._input_noise).numpy()
        X['trial_start'] = np.tile(
            self._trial_start, (n * len(self._trial_types),))
        X['light'] = X['trial_start'] + \
                            tf.random.normal((len(X['cues']),), mean=0,
                                             stddev=self._input_noise).numpy()
        X['response'] = trial_list.apply(
            lambda x: self.response_window.loc[x]).stack().values
        X = X.reindex(['trial_start', 'light', 'cues', 'odor', 'response'], axis=1)
        return X


class LSOnlyTask(tDNMSGenerator):
    def __init__(self, **kwargs):
        super(LSOnlyTask, self).__init__(
            trial_block=('LS', 'LS', 'LS', 'LS'), **kwargs)


class SLOnlyTask(tDNMSGenerator):
    def __init__(self, **kwargs):
        super(SLOnlyTask, self).__init__(
            trial_block=('SL', 'SL', 'SL', 'SL'), **kwargs)


class MMOnlyTask(TrialGenerator):
    def __init__(self, **kwargs):
        mm = [0] * 30 + [1] * 35 + [0] * 30 + [1] * 35
        cues = {'MM': mm}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'MM': response}
        trial_blocks = ('MM', 'MM', 'MM', 'MM')

        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_blocks, **kwargs)


class tDNMS_no_match(TrialGenerator):
    def __init__(self, **kwargs):
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=('SL', 'SL', 'LS', 'LS'), **kwargs)


class tDNMS_no_match_3input(TrialGenerator):
    def __init__(self, **kwargs):
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        no_cue = [0] * 130
        cues = {'SL': np.array([sl, no_cue, no_cue]), 'LS': np.array([ls, no_cue, no_cue])}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=('SL', 'SL', 'LS', 'LS'), n_cues=3, **kwargs)


class tDNMS_LongISI(TrialGenerator):
    def __init__(self, **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 50 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 50 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 50 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(170)
        response[151:170] = 1
        targets = {'SL': response, 'LS': response}
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=('SS', 'SL', 'LS', 'SS'), **kwargs)


class tDNMSGeneratorLL(TrialGenerator):
    def __init__(self, keep_SS=False, trial_block=None, **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        ll = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 50
        cues = {'SS': ss, 'SL': sl, 'LS': ls, 'LL': ll}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        if trial_block is None:
            trial_blocks = trial_block
        elif keep_SS:
            trial_blocks = ('LL', 'SL', 'LS', 'SS')
        else:
            trial_blocks = ('LL', 'SL', 'LS', 'LL')

        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_blocks, **kwargs)


class CueResponse(TrialGenerator):
    def __init__(self, offset=True, cue_length=2, **kwargs):
        sampling_rate = 0.1
        cued_bins = int(cue_length / sampling_rate)
        s_only = [0] * 30 + [1] * cued_bins
        cues = {'S': s_only}
        trial_blocks = ('S', 'S', 'S', 'S')

        response = np.zeros(150)
        if offset:
            response[(30 + cued_bins):(30 + cued_bins + 20)] = 1
        else:
            response[31:(30 + cued_bins)] = 1

        targets = {'S': response}

        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=trial_blocks, **kwargs)
        
class tDNMSRespAll(TrialGenerator):
    def __init__(self, **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response, 'SS': response} 
        super(self.__class__, self).__init__(
            cues, targets, 200, trial_blocks=('SS', 'SL', 'LS', 'SS'), **kwargs)

class genFactory:
    @classmethod
    def create(cls, task, input_noise, batch_size, n_blocks):
        if task == 'just_short_match' or task == 'potentiate':
            traingen = tDNMSGenerator(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'just_short_match_3input':
            traingen = tDNMSGenerator_3input(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'include_long_match':
            traingen = tDNMSGeneratorLL(
                input_noise=input_noise, keep_SS=True, batch_size=2, n_blocks=n_blocks)
        elif task == 'sl_only':
            traingen = SLOnlyTask(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'ls_only':
            traingen = LSOnlyTask(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'mm_only':
            traingen = MMOnlyTask(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'no_match':
            traingen = tDNMS_no_match(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'no_match_3input':
            traingen = tDNMS_no_match_3input(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'cue_response':
            traingen = CueResponse(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'long_cue_response':
            traingen = CueResponse(input_noise=input_noise, batch_size=2, n_blocks=n_blocks, cue_length=10)
        elif task == 'cue_response_simul':
            traingen = CueResponse(offset=False, input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'long_isi':
            traingen = tDNMS_LongISI(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'two_cue_dnms':
            traingen = DNMS2CueGenerator(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'two_cue_dnms_3input':
            traingen = DNMS2CueGenerator_3input(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'respond_all':
            traingen = tDNMSRespAll(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'super_long':
            traingen = SuperLongTrial(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'include_med_match':
            traingen = tDNMSGeneratorMM(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'omit_one':
            traingen = OmitOne(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'temporal_jitter':
            traingen = tDNMSTemporalJitter(input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
        elif task == 'match_only':
            traingen = tDNMSGeneratorLL(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks, trial_block=('SS', 'LL'))
        else:
            raise Exception('task not found')

        return traingen
