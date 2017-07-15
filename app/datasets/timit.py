import app.hparams as hparams
from app.datasets.dataset import Dataset


@hparams.register_dataset('timit')
class TimitDataset(Dataset):
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size):
        # TODO
        pass

    def install_and_load(self):
        # TODO
        pass
