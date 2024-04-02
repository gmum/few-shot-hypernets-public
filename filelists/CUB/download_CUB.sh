#!/usr/bin/env bash
wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py
