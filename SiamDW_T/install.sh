pip install torch==0.3.1
pip install cffi
pip install opencv-python
pip install torchvision==0.2.1
pip install pytorch_fft
pip install shapely

echo "****************** Installing PreROIPooling ******************"
base_dir=$(pwd)
cd libs/models/external/PreciseRoIPooling/pytorch/prroi_pool
PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh
cd $base_dir

echo ""
echo "****************** Installation complete! ******************"
