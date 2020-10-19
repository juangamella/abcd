#!/usr/bin/env bash

python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
#pip3 install --no-cache-dir --upgrade causaldag # AICP Paper: Causaldag's newer versions break the dependency tree

