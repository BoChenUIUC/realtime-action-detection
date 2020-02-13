#!/usr/bin/env bash
mkdir /home/monet/research/dataset/jhmdb -p
cd /home/monet/research/dataset/jhmdb
wget http://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz
tar -xvzf Rename_Images.tar.gz
wget http://files.is.tue.mpg.de/jhmdb/puppet_mask.zip
unzip puppet_mask.zip
wget http://files.is.tue.mpg.de/jhmdb/splits.zip
unzip splits.zip

mkdir cache