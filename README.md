# active_learning

Forked from [agrawalraj/active_learning](https://github.com/agrawalraj/active_learning).

You will need at least

Python 3.6
R 3.6

The installation procedure they provide didn't work for me, and some python dependencies were missing from the venv. This is what I did to get the code to run.

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
cd new
bash make_venv.sh
source venv/bin/activate
pip install pyaml tqdm xarray causaldag
```

### Executing experiments

Either create a dataset or load it.