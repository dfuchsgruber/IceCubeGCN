
#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
source ~/myenv2/bin/activate
cd ~/IceCubeGCN
python dataset/create.py $1 ~/data/all_energies/$1.hd5 100