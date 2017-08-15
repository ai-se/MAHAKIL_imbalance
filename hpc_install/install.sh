#!/usr/bin/env bash

#cd /share2/aagrawa8/

#wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh

#bash Miniconda2-latest-Linux-x86_64.sh # take care of the installation directory

#-q standard -W 5000
#-q shared_memory -W 10000
#-q long -W 20000

#cd
conda remove mkl mkl-service -y
conda install nomkl -y
conda install pip numpy scipy scikit-learn matplotlib -y
#conda install mkl -y
pip install -U lda nltk
pip install pandas

#python
#import nltk
#nltk.download
