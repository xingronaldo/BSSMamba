# Code for manuscript 'Change Detection Mamba with Boundary-Specific Supervision'.
---------------------------------------------
Here I provide the PyTorch implementation for BSSMamba.


## ENVIRONMENT
>RTX 3090<br>
>python 3.8<br>
>PyTorch 2.0.0<br>
>mmcv-full 1.6.0<br>
>causal_conv1d 1.1.0<br>
>mamba_ssm 1.1.1

## Installation
Clone this repo:

```shell
git clone https://github.com/xingronaldo/BSSMamba.git
cd BSSMamba
```

* Install dependencies

All dependencies can be installed via 'pip'.

## Dataset Preparation
Download data and add them to `./datasets`. 

## Train & Validation
```python
python trainval.py --gpu_ids 1 --name LEVIR
```
All the hyperparameters can be adjusted in `option.py`.


## TODO
A more detailed README.

## Contact
Email: guangxingwang@mail.nwpu.edu.cn





