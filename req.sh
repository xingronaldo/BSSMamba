#!/bin/bash

echo "start installing dependencies..."

pip install einops
pip install kornia
pip install timm==0.6.13
pip install psutil ninja
apt-get update -y
apt-get install libglib2.0-0 -y

echo "Finished"
