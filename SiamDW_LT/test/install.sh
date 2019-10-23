#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.6.0 ******************"
conda create -y --name $conda_env_name python=3.6.0

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch 0.3.1 ******************"
pip install torch==0.3.1


echo ""
echo ""
echo "****************** Installing matplotlib 2.2.2 ******************"
conda install -y matplotlib=2.2.2

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python==3.1.0.5

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing numpy==1.12.1 ******************"
pip install numpy==1.12.1

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing pytorch_fft ******************"
pip install pytorch_fft

echo ""
echo ""
echo "****************** Installing pytorch_fft ******************"
pip install torchvision==0.2.2

echo ""
echo ""
echo "****************** Installing PreROIPooling ******************"
base_dir=$(pwd)
cd libs/PreciseRoIPooling/pytorch/prroi_pool
PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh
cd $base_dir

echo ""
echo ""
echo "****************** Compiling Trax library ******************"
cd native/trax
mkdir build
cd build
cmake ..
make

echo ""
echo ""
echo "****************** Compiling Trax Python ******************"
cd $base_dir
cd native/trax/support/python
mkdir build
cd build
cmake ..
make

echo ""
echo ""
echo "****************** Installation complete! ******************"
