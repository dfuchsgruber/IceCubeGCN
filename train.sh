#!/bin/bash

cd ~/IceCubeGCN
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
source ~/myenv3/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'
export ICECUBEGCNMEMMAPDIR='/data/user/dfuchsgruber/memmaps'
python3 train.py settings/hd5.json

