import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
import torch

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, epoch=60):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (ptbtir, lsotb_conftest, lashertir, rgbt234tir, lsotb_st, lsotb_lt).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)
    if epoch is not None:
        run_id = run_id + f"_ep{epoch}"

    if sequence is not None:
        if isinstance(sequence, list):
            dataset = [dataset[seq] for seq in sequence]
        else:
            dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, epoch=epoch)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=str, default=None, help='The run name.')
    parser.add_argument('--dataset_name', type=str, default='lashertir', help='Name of dataset (ptbtir, lsotb_all, lashertir, vtuavtir).')
    parser.add_argument('--sequence', nargs='+', help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--epoch', type=int, default=60, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, epoch=args.epoch)


if __name__ == '__main__':
    main()
