#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.0/setup.sh`
source /cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_7_x86_64/metaprojects/simulation/V06-00-00/env-shell.sh
source ~/myenv2/bin/activate
cd ~/IceCubeGCN
echo asdf
python dataset/create.py 0 ~/data/all_energies/test.hd5 2
echo asdf
