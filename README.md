# 3DAunet
#Overview
Code for TIM paper:** 3D Particle Reconstruction for Flow Motion Estimation via Multi-Scale Pyramid Network**

# System Requirements
- operating system: The training and testing of the code are in Windows 11
- software dependencies:
 -Python 3.11.9
 -CUDA 12.5
 -Hardware: tested and trained on a single NVIDIA RTX3090, with RAM 24G
 -Python dependencies:
 -PyTorch 2.4.0
 -tensorboard 2.17.0
 -numpy 1.26.4
 -scipy 1.14.1
 -pandas 2.2.3
 -pillow 10.4.0

# Installation Guide
The code has been tested with Python 3.11.9, PyTorch 2.4.0, CUDA 12.5 onWindows 11 24H2.

If you want to clone this repository：
```
git clone https://github.com/Painkllller/3DAunet.git
cd 3DAunet/
```
#Install pytorch and other python dependencies:
```
pip install torch==2.4.0
pip install tensorboard==2.17.0
pip install numpy==1.26.4
pip install scipy==1.14.1
pip install pandas==2.2.3
```

# Required Data
If you want to evaluate and test the model presented in this article, you can access (https://drive.google.com/drive/folders/1n0e7Os0_2wrU8wHfX5AOCHaKZ1ZGCMWa?usp=drive_link) to obtain the weights of the model and a validation set with a particle concentration of 0.3 for testing. If your running result is around 0.75, it indicates that your steps are correct.
Obtain the complete data set,You can access the project(https://github.com/Painkllller/The-dataset-production-of-MLOS-SF-MART.git) in Matlab. Open the project running MLOS.mat files making training set, In the paper, 240 particle fields of 256*256*128 were fabricated and divided into a total of 15,360 small particle fields of 64*64*32 as the training set. You can create your own dataset according to the situation.

# Train & Test
-Before running train.py, run makecvs.py to preprocess the data first.
-The model weights are saved in early_stopping.py.
-Run test.py you can get the aerage Q value of the data（small field）, If you want to obtain the q value of the large field after stitching, test. Save the running results of test.py and then run reconstruct. Py
