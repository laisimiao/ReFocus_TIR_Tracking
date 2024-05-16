from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    lsotb_st=DatasetInfo(module=pt % "lsotb", class_name="LSOTBDataset", kwargs=dict(split='ST')),
    lsotb_lt=DatasetInfo(module=pt % "lsotb", class_name="LSOTBDataset", kwargs=dict(split='LT')),
    lsotb_conftest=DatasetInfo(module=pt % "lsotb", class_name="LSOTBDataset", kwargs=dict(split='ConfTest')),
    lsotb_all=DatasetInfo(module=pt % "lsotb", class_name="LSOTBDataset", kwargs=dict(split='ALL')),
    lashertir=DatasetInfo(module=pt % "lashertir", class_name="LasHerTIRDataset", kwargs=dict()),
    ptbtir=DatasetInfo(module=pt % "ptbtir", class_name="PTBTIRDataset", kwargs=dict()),
    vtuavtir=DatasetInfo(module=pt % "vtuavtir", class_name="VTUAVTIRDataset", kwargs=dict(subset='st')),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset