#!/bin/bash
# init common pkgs for Py3

# sudo apt-get install python3-pip

# sudo pip3 install jupyter
# sudo pip3 install numpy
# sudo pip3 install opencv-python

# sudo apt-get install libpng-dev
# sudo apt-get install libfreetype6-dev
# pip3 install matplotlib
# sudo apt install libsm6 libxext6 libxrender1


# install cuda-8.0

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-8-0; then
# The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda-8-0 -y
fi


# install torch, torchvision

sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
sudo pip3 install torchvision

# install jupyter kernel
sudo python2 -m pip install ipykernel
sudo python2 -m ipykernel install --user