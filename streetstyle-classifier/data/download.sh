#!/bin/bash

# download streetstyle-27k dataset
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.manifest
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.tar
tar -xvf streetstyle27k.tar

# download NewsAnchor_train dataset
wget https://storage.googleapis.com/esper/Haotian/newsAnchor_train_img.tar.gz
wget https://storage.googleapis.com/esper/Haotian/newsAnchor_train_manifest.pkl
tar -xvf newsAnchor_train_img.tar.gz

# download NewsAnchor dataset
wget https://storage.googleapis.com/esper/Haotian/newsAnchor_all_img.tar.gz
wget https://storage.googleapis.com/esper/Haotian/newsAnchor_all_manifest.pkl
tar -xvf newsAnchor_all_img.tar.gz
