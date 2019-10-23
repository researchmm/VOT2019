import os

os.system('rm -f *.so')
os.system('python setup.py build_ext --inplace')
