import os
import itertools as itr

ns = [24, 48, 96, 192]
ks = [None]
bs = [1, 2, 3]

p = 25
s = .25
ndags = 50

os.system(f'python3 make_dataset.py -p {p} -s {.25} -d {ndags} -t erdos-bounded --folder erdos_renyi_descendant')
for n, b, k in itr.product(ns, bs, ks):
    if k is None:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy entropy-dag-collection-descendants')
    else:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy entropy-dag-collection-descendants')
for n, b, k in itr.product(ns, bs, ks):
    if k is None:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy entropy-dag-collection')
    else:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy entropy-dag-collection')
for n, b, k in itr.product(ns, bs, ks):
    if k is None:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy random-smart')
    else:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant --strategy random-smart')
