#!/bin/bash

# If on centos/7, we will need the following
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python python-devel
sudo yum install -y epel-release
sudo yum install -y python-pip
sudo pip install --upgrade pip
sudo yum install -y wget

# If planning to use virtual environment
pip install --upgrade --user virtualenv
virtualenv . -p /usr/bin/python2.7

# get FFHT repo
wget https://github.com/FALCONN-LIB/FFHT/archive/master.zip
unzip master.zip

# get necessary tools
pip install --upgrade numpy --user
pip install --upgrade scipy --user
pip install --upgrade pandas --user
pip install --upgrade matplotlib --user
pip install --upgrade scikit-learn --user
pip install --upgrade tqdm --user
pip install --upgrade h5py --user

# install FFHT
cd FFHT-master
pip install . --user
# python setup.py install --user

# git clone git@github.com:FALCONN-LIB/FFHT.git
# cd FFHT
# pip install . --user
# python setup.py install

