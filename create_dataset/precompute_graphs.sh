#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
source ~/myenv3/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'
cd ~/IceCubeGCN

cp /data/user/dfuchsgruber/all_energies/train.hd5 /data/user/dfuchsgruber/all_energies/train_precomputed.hd5
python3 create_dataset/precompute_graphs.py /data/user/dfuchsgruber/all_energies/train_precomputed.hd5 50

cp /data/user/dfuchsgruber/all_energies/val.hd5 /data/user/dfuchsgruber/all_energies/val_precomputed.hd5
python3 create_dataset/precompute_graphs.py /data/user/dfuchsgruber/all_energies/val_precomputed.hd5 50

cp /data/user/dfuchsgruber/all_energies/test.hd5 /data/user/dfuchsgruber/all_energies/test_precomputed.hd5
python3 create_dataset/precompute_graphs.py /data/user/dfuchsgruber/all_energies/test_precomputed.hd5 50
