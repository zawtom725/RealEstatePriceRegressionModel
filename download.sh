#!/usr/bin/env bash

pip list | grep kaggle > /dev/null

if [ $? -eq 0 ]; then
    echo "kaggle installed"
else
    pip install kaggle
fi
# maybe ask for account json?
ls -d */ | grep data > /dev/null
if [ $? -ne 0 ]; then
    mkdir data
    kaggle competitions download -c zillow-prize-1 -p data
else
    echo "data is downloaded"
fi
