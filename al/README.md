# README

**Note: These scripts require an additional requirement that the
main QuasarNET/NP repos do not: [SOMVIZ](https://github.com/belaa/SOMVIZ)**

These scripts and the one notebook provide everything needed to run 
the QuasarNET active learning pipeline, end to end. The active 
learning pipeline consists of three main steps:

1. Training an ensemble of QuasarNET classifiers.
2. Running non-training data through the trained ensemble
to generate confidence values.
3. Determining which input data/spectra would most benefit the 
network to be labeled and included in the training dataset.



## Training

Training is done using `train_qnet_ensemble.py`.

This script is similar in construction to [James Farr's 
train_quasarnet scripts.](https://github.com/dylanagreen/QuasarNET/blob/master/bin/qn_train)
Options are provided to fine tune QuasarNET structure as well as
what lines to train for and how many epochs to train each network
in the ensemble for. Note that the entire ensemble will be trained 
identically except for the dataset, which is bootstrapped from the
input datset. 
There is no current funcionality to independently change
the structure of individual ensemble members. 

`--data` parameter is designed to take in eBOSS data to supplement the
DESI data passed in via `--desidata`. Two parameters are included 
that are not included in quasarnet training scripts. 
The first `--num_bootstrap` which controls
how many networks are in the ensemble and defaults to 100. 
The second is `--single_exp` which only uses a single spectra
/exposure for each TARGETID rather than the entirety of the datset
that was passed in.

### Example:

`./train_qnet_ensemble.py --amsgrad -t "./training_data/truth_dr12q.fits" -d "./training_data/dr12_train.fits" -o "." -dd "./fuji_data/fuji_perexp_combined_QSO_truthmatched.fits" -dt "./fuji_data/desi_vi_blanc_truth_QSO_combined.csv" -n 200 -e 10`

Here's a breakdown of the parameters in the above run:
- `-dd` and `-dt`: Train on internal DESI fuji data.
- `-d` and `-t`: Supplement DESI data with public dr12 eBOSS data.
- `--amsgrad`: Use the amsgrad algorithm for the training optimizer.
- `-o`: Create the output ensemble weights folder in the current working directory
- `-n`: Use 200 networks to create the ensemble.
- `-e`: Train each ensemble member for 10 epochs .


## Running

Running data through the ensemble is done through either `ensemble_run.py`
or `ensemble_run_tf.py`. The latter is identical to the former, 
except that instead of using QuasarNP (with no TensorFlow
requirement) it uses QuasarNET (which does require TensorFlow). 
This can be beneficial in environments with GPU access, as QuasarNET
can be up to 10x faster than the numpy only QuasarNP.

This script will load the ensemble from the location provided by `--ensemble_loc` which
should be the same directory passed to `train_qnet_ensemble.py` in that script's
`--out-dir` argument. This script has its own `--out-dir` which defines the location
to save the csv file with the ensemble's results. 

### Example:
`./ensemble_run_tf.py -d "$GUADALUPE_LOC" -o ".." -i "../ensemble_cumulative"`

Here's a breakdown:
- `-d`: Run the internal DESI guadalupe data through the ensemble,
location redacted for public release of this README.
- `-o`: Save the output csv in the encompassing parent folder
(so that it doesn't save it in the scripts directory).
- `-i`: Use the ensemble weights saved in `ensemble_cumulative`
folder saved in the parent directory.

## Querying
Querying right now is done through the jupyter notebook 
`al_querier.ipynb`. The notebook should be self exaplanatory, and 
the best way to see how it works is to read the markdown cells in
the notebook.

Future work *may* extract this into a script, but it's so short and
the visualizations provide enough benefit that this may not be
necessary.
