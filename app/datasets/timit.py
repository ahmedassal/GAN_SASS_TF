import os
from six.moves import cPickle as pickle

import numpy as np

import app.hparams as hparams
from app.datasets.dataset import Dataset

# TODO should we use pathlib to handle path?
#      (also means dropping python2 support)


@hparams.register_dataset('timit')
class TimitDataset(Dataset):
    '''TIMIT dataset'''
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size):
        # TODO add a validation set ?
        if subset not in self.subset:
            raise KeyError(
                'Unknown subset "%s", valid options are %s' %
                (subset, list(self.subset.keys())))
        signals_li, phonemes_li, texts_li = self.subset[subset]
        tot_size = len(signals_li)
        assert tot_size == len(phonemes_li)
        assert tot_size == len(texts_li)
        for i in range(0, tot_size-batch_size, batch_size):
            sig_len = len(signals_li[i+batch_size-1])
            txt_len = max(map(len, texts_li[i:i+batch_size]))
            signals = np.stack(
                [np.pad(s, ((0, sig_len-len(s)), (0, 0)), mode='constant')
                    for s in signals_li[i:i+batch_size]])
            texts = np.stack(
                [np.pad(t, ((0, txt_len-len(t)),), mode='constant')
                    for t in texts_li[i:i+batch_size]])
            yield signals, texts
        if tot_size % batch_size:
            sig_len = len(signals_li[-1])
            txt_len = max(map(len, texts_li[i:i+batch_size]))
            signals = np.stack(
                [np.pad(s, ((0, sig_len-len(s)), (0, 0)), mode='constant')
                    for s in signals_li[-batch_size:]])
            texts = np.stack(
                [np.pad(t, ((0, txt_len-len(t)),), mode='constant')
                    for t in texts_li[-batch_size]])
            yield signals, texts

    def install_and_load(self):
        # TODO automatically install if fails to find anything
        FILE_NOT_FOUND_MSG = (
            'Did not found TIMIT %s file'
            ', make sure you download and install the dataset')
        self.subset = {}
        path = os.path.join(os.path.dirname(__file__), 'TIMIT', '%s_set.pkl')
        for subset in ['train', 'test']:
            filepath = path % subset
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                        FILE_NOT_FOUND_MSG % filepath)

            with open(filepath, 'rb') as f:
                all_data = [pickle.load(f)]
                all_data.append(pickle.load(f))
                all_data.append(pickle.load(f))
            self.subset[subset] = all_data
