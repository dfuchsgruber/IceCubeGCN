#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
source ~/myenv3/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'
cd ~/IceCubeGCN
python3 create_dataset/concatenate_datasets.py "/data/user/dfuchsgruber/all_energies/*.hd5" /data/user/dfuchsgruber/all_energies.hd5 1000000
