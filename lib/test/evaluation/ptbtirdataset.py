import numpy as np
import json
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from glob import glob


class PTBTIRDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.ptbtir_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        ground_truth_rect = np.loadtxt(os.path.join(self.base_path, sequence_name, 'groundtruth_rect.txt'), delimiter=',', dtype=np.float32)
        frames_list = sorted(glob(os.path.join(self.base_path, sequence_name, 'img', '*.jpg')))

        return Sequence(sequence_name, frames_list, 'ptbtir', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [i for i in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, i))]
        return sequence_list
