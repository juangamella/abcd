#!/usr/bin/env bash

python3 make_dataset.py -p 15 -d 1 -t chain_one_direction --folder chain_test10

python3 run_experiments.py -n 150 -b 3 -k 1 -m 10 -s .1 -i gauss --verbose 1 --folder chain_test10 --strategy entropy-dag-collection

