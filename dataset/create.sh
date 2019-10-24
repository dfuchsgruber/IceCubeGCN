#!/bin/bash

echo $1
source ~/myenv2/bin/activate
python dataset/create.py $1 /data/user/dfuchsgruber/all_energies/$1.hd5 1
