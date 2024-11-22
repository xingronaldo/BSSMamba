#!/bin/bash

echo "start installing dependencies..."

pip install einops
pip install kornia
pip install timm==0.6.13
pip install psutil ninja
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
apt-get update -y
apt-get install libglib2.0-0 -y

echo "Finished"
