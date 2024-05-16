import numpy as np
import json
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LSOTBDataset(BaseDataset):
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.lsotb_path
        self.sequence_list = self._get_sequence_list(split)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):

        ground_truth_rect = np.array(self.meta[sequence_name]["gt_rect"], dtype=np.float32)
        frames_list = [os.path.join(self.base_path, i) for i in self.meta[sequence_name]["img_names"]]

        return Sequence(sequence_name, frames_list, 'lsotb', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        if split == "ST":
            test_json = os.path.join(self.base_path, 'LSOTB-TIR-ST100.json')
        elif split == 'LT':
            test_json = os.path.join(self.base_path, 'LSOTB-TIR-LT11.json')
        elif split == 'ConfTest':
            test_json = os.path.join(self.base_path, 'LSOTB-TIR-120.json')
        elif split == 'ALL':
            test_json = os.path.join(self.base_path, 'LSOTB-TIR-136.json')
        else:
            raise ValueError(f"Now only support ST/LT/ConfTest, but got {split}")

        with open(test_json, 'r') as f:
            self.meta = json.load(f)
        sequence_list = sorted(list(self.meta.keys()))
        return sequence_list
