# dependencies
apt-get update
apt-get install python3-dev
pip install --upgrade pip
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# pytorch
cd /opt/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

# ruinmessi/RFBNet
apt-get install python3-tk
pip3 install opencv-python cython matplotlib

