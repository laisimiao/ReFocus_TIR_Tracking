import numpy as np
import os
from glob import glob
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LasHerTIRDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lashertir_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/infrared.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/infrared/'.format(self.base_path, sequence_name)

        frames_list = sorted(glob(os.path.join(frames_path, '*.jpg')))

        return Sequence(sequence_name, frames_list, 'lashertir', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open(os.path.join(self.base_path, 'testingsetList.txt'), 'r') as f:
            seqs = f.read().splitlines()
        sequence_list = sorted(seqs)
        return sequence_list
