# 3Dregistration-pcrnet

## Setup
Use this command to install the required packages. 
```
pip install -r requirements.txt
```

If you want to use EMDLoss, you need to install the following modules.
```
git clone https://github.com/yym064/EMDLoss_PyTorch_cpp_extension.git
cd EMDLoss_PyTorch_cpp_extension
python setup.py install
```

## References:
1. [PCRNet](https://arxiv.org/abs/1612.00593): Point Cloud Registration Network using PointNet Encoding.
2. [Learning3D](https://github.com/vinits5/learning3d): A Modern Library for Deep Learning on 3D Point Clouds Data.
3. [EMDLoss](https://github.com/yym064/EMDLoss_PyTorch_cpp_extension): EMDLoss PyTorch cpp extension.