#!/bin/bash

echo $1
cd ~/IceCubeGCN
#source ~/myenv2/bin/activate
export PYTONPATH=$PYTHONPATH:~/myenv2/lib/python2.7/site-packages/
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.0/setup.sh`
source /cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_7_x86_64/metaprojects/simulation/V06-00-00/env-shell.sh dataset/create.sh $1

