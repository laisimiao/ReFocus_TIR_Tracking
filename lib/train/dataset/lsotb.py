import os
import os.path
import numpy as np
import torch
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

class LSOTB(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lsotb_dir if root is None else root
        assert split in ['train', 'val'], 'Only support train or val split in LSOTB, but got {}'.format(split)
        super().__init__('LSOTB', root, image_loader)
        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'LSOTB'

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lsotb_{}_split.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        ir_bb_anno_file = os.path.join(seq_path.replace("TrainingData", "MyAnnotations"), "tir.txt")  # xywh
        ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(ir_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, 'TrainingData', self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        ir_frame_path = os.path.join(seq_path, '{:08d}.jpg'.format(frame_id+1))
        return ir_frame_path

    def _get_frame(self, seq_path, frame_id):
        img = self.image_loader(self._get_frame_path(seq_path, frame_id))
        return img 

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta