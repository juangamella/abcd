# ABCD vs A-ICP comparison

This repository contains the code to reproduce the results comparing [ABCD](https://arxiv.org/abs/1902.10347) to [A-ICP](https://github.com/juangamella/aicp) in the paper *Active Invariant Causal Prediction: Experiment Selection through Stability*, by Juan L Gamella and Christina Heinze-Deml.

The repository is forked from [agrawalraj/active_learning](https://github.com/agrawalraj/active_learning), which contains the original implementation of [ABCD](https://arxiv.org/abs/1902.10347). It contains minor changes to the code to get it to run and retrieve results for the experiments comparing ABCD to A-ICP. Changes to the original code are marked with a comment: `#A-ICP paper: *`.

## Dependencies

You will need at least

Python 3.6

R 3.6

The original installation procedure didn't work for us, and some python dependencies were missing from the venv. This is what we did to get the code to run.

### Installing the R dependencies:

In an R terminal (also in `install.R`)

```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install()
BiocManager::install("Rgraphviz, RBGL")
install.packages('pcalg', repos='http://cran.us.r-project.org')
install.packages('gRbase', repos='http://cran.us.r-project.org')
```

### Installing the Python dependencies

```
cd new/
bash make_venv.sh
source venv/bin/activate
pip install pyaml tqdm xarray causaldag
```

## Reproducing the results

The dataset used to run the experiments is generated through the code in the A-ICP [implementation](https://github.com/juangamella/aicp) (see *Reproducing experiments* in the README), and then copied to the `new/data/` directory. The dataset is a directory structure (here `dataset/`). Unfortunately, running the code renders the dataset unusable for other runs, so we have to copy it (I like to keep `dataset/` as the "master copy"). For the experiments (a total of 12), we copy it 12 times, plus one to test everything works:

```
cd data/
cp -r dataset dataset_0
cp -r dataset dataset_1
cp -r dataset dataset_2
cp -r dataset dataset_3
cp -r dataset dataset_4
cp -r dataset dataset_5
cp -r dataset dataset_6
cp -r dataset dataset_7
cp -r dataset dataset_8
cp -r dataset dataset_9
cp -r dataset dataset_10
cp -r dataset dataset_11
cp -r dataset dataset_test
```

I would then run a small experiment once to see if everything works:

```
python run_experiments.py -n 10 -b 2 -k 1 --boot 20 -s 7 --folder dataset_test --strategy entropy --starting-samples 100
```

If it doesn't crash after 1 minute, kill it and run the rest. To run the ABCD algorithm at 50, 100 and 1000 observational sample sizes, each for four rounds:

```
cd new/

python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_0 --strategy entropy --starting-samples 50
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_1 --strategy entropy --starting-samples 50
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_2 --strategy entropy --starting-samples 50
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_3 --strategy entropy --starting-samples 50

python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_4 --strategy entropy --starting-samples 100
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_5 --strategy entropy --starting-samples 100
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_6 --strategy entropy --starting-samples 100
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_7 --strategy entropy --starting-samples 100

python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_8 --strategy entropy --starting-samples 1000
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_9 --strategy entropy --starting-samples 1000
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_10 --strategy entropy --starting-samples 1000
python run_experiments.py -n 500 -b 50 -k 1 --boot 100 -s 7 --folder dataset_11 --strategy entropy --starting-samples 1000
```

**Parallelization**

The code automatically runs on as many cores as are made available to it, minus one.

**Results**

The results are pickled into the `new` directory. The filenames contain a timestamp and the parameters used, eg.

```
pp_1589260274_n_samples:500_n_batches:50_max_interventions:1_strategy:entropy_intervention_strength:5.0_starting_samples:100_target:0_intervention_type:gauss_target_allowed:True.pickle
```

They can be plotted with the `plots_abcd.ipynb` notebook in the A-ICP [repository](https://github.com/juangamella/aicp).
