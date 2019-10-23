conda install -y cython
pip install multiprocess
pip install torch==0.3.1
pip install numpy==1.16.4
pip install matplotlib
pip install cffi
pip install opencv-python
pip install torchvision==0.2.0
pip install pytorch_fft

base_dir=$(pwd)
cd libs/PreciseRoIPooling/pytorch/prroi_pool
PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh
cd $base_dir

cd libs/FPNlib/
sh install.sh
