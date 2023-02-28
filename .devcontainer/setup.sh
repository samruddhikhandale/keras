#!/bin/bash

set -eux
sudo pip install -r requirements.txt
sudo pip uninstall keras-nightly -y
