#!/usr/bin/env python3

"""
Trains an ensemble of QuasarNET classifiers on the given data.

This script is similar in construction to James Farr's train_quasarnet
scripts. Options are provided to fine tune QuasarNET structure as well as
what lines to train for and how many epochs to train each network
in the ensemble for. Note that the entire ensemble will be trained 
identically except for the dataest, which is bootstrapped from the
input datset. There is no current funcionality to independently change
the structure of individual ensemble members. 

`--data` parameter is designed to take in eBOSS data to supplement the
DESI data passed in via `--desidata`. Two parameters are included 
that are not included in quasarnet training scripts. 
The first `--num_bootstrap` which controls
how many networks are in the ensemble and defaults to 100. The second
is `--single_exp` which only uses a single spectra/exposure for each 
TARGETID rather than the entirety of the datset that was passed in.

"""


# Extensive amount of imports...
from quasarnet.models import QuasarNET, custom_loss
from quasarnet import io
from quasarnp.utils import process_preds, absorber_IGM, wave, rebin, renormalize
from quasarnp.io import load_desi_coadd

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.layer_utils import count_params
import fitsio

import argparse
import os
import time
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("-t", "--truth", type=str, required=False, nargs="*")
parser.add_argument("-d", "--data", type=str, required=True, nargs="*")

parser.add_argument("-dt", "--desitruth", type=str, required=False, nargs="*")
parser.add_argument("-dd", "--desidata", type=str, required=False, nargs="*")

parser.add_argument("-e", "--epochs", type=int, required=False, default=8)
parser.add_argument("--boxes", type=int, required=False, default=13)
parser.add_argument("-o", "--out-prefix", type=str, required=True)

default_lines = ["LYA", "CIV(1548)", "CIII(1909)", "MgII(2796)", "Hbeta", "Halpha"]
parser.add_argument("-l", "--lines", type=str, required=False,
                    default=default_lines, nargs="*")
parser.add_argument("-b", "--lines-bal", type=str, required=False,
                    default=["CIV(1548)"], nargs="*")

# parser.add_argument("--offset-activation-function", type=str,
#                     choices=['sigmoid', 'rescaled_sigmoid', 'linear'],
#                     default='rescaled_sigmoid', required=False)
parser.add_argument("--amsgrad", action='store_true', default=False,
                    required=False)
# parser.add_argument("--no-shuffle", action='store_true', default=False,
#                     required=False)
parser.add_argument("--seed", type=int, required=False, default=91701)
parser.add_argument("-n", "--num-bootstrap", type=int, required=False, default=100)
parser.add_argument("-se", "--single-exp", required=False, action="store_true")

args = parser.parse_args()

# Reproducibility variables.
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = '1'

rng = np.random.default_rng(args.seed)


# Where to save data and load data
## TODO: Make this a script var
cwd = Path.cwd()
# d_loc = cwd.parents[0] / "qnp" / "training_data"
out_loc = cwd / args.out_prefix
print(out_loc)

if not os.path.isdir(out_loc):
    os.mkdir(out_loc)

weights_loc = out_loc / "ensemble_se" if args.single_exp else out_loc / "ensemble" 

# For training on coadds instead of individual exposures.
if "cumulative" in args.desidata[0]: weights_loc = out_loc / "ensemble_cumulative" 
if "healpix" in args.desidata[0]: weights_loc = out_loc / "ensemble_healpix" 

if not os.path.isdir(weights_loc):
    os.mkdir(weights_loc)


include_dr12 = True

tids_d, X_d, Y_d, z_d, bal_d = io.read_data_desi(args.desidata[0], args.desitruth[0])

if args.single_exp:
    # Shuffling ensures we return a random choice of spectra for each TID
    # instead of just the first when we use np.unique
    inds = np.arange(len(tids_d))
    rng.shuffle(inds)
    _, desi_labeled = np.unique(tids_d[inds], return_index=True)
    
    X_d = X_d[inds][desi_labeled]
    Y_d = Y_d[inds][desi_labeled]
    z_d = z_d[inds][desi_labeled]
    bal_d = bal_d[inds][desi_labeled]

print(f"{len(z_d)} desi training samples.")
n_desi = X_d.shape[0]
 
    
if include_dr12:
    truth = io.read_truth(args.truth)
    tids, X, Y, z, bal = io.read_data_boss(args.data, truth, load_photo=False)
    
    if len(z) > len(z_d):
        dr12_labeled = rng.choice(np.arange(len(z)), size=4*4000, replace=False)
    else:
        dr12_labeled = np.arange(len(z))
    n_dr12 = len(z)
    
    print(f"{n_dr12} labeled indices from DR12")

    # I recognize that empty : does nothing but I don't want to have to 
    # type them back up when I want to reduce the amount of DR12 data again.
    X = np.concatenate([X[dr12_labeled][:, :], X_d], axis=0)
    Y = np.concatenate([Y[dr12_labeled][:, :], Y_d], axis=0)
    z = np.concatenate([z[dr12_labeled], z_d], axis=0)
    bal = np.concatenate([bal[dr12_labeled], bal_d], axis=0)

print(f"{len(z)} total training samples")

lmin = 3600.
lmax = 10000.
dll = 1e-3 #/ 2
box, sample_weight = io.objective(z, Y, bal, lines=args.lines,
                                  lines_bal=args.lines_bal, nboxes=args.boxes,
                                  llmin=np.log10(lmin),
                                  llmax=np.log10(lmax), dll=dll)

# Training parameters
# In single exposure training, dividing by 10 was a better learning rate.
# I'm using twice that here since we only train for 8 epochs, to get a good sense
# of where it would end up just a little faster.
initial_lr = 0.001 / 5 
offset_activation = "rescaled_sigmoid"
adam = Adam(learning_rate=initial_lr, decay=0.00, amsgrad=args.amsgrad)
loss = []
for i in args.lines:
    loss.append(custom_loss)
for i in args.lines_bal:
    loss.append(custom_loss)
    

# For bootstrapping selections
inds = np.arange(len(X))

# This is the core loop that bootstraps an ensemble and trains on a 
# bootstrapped subset of the input data. Each loop it chooses the same amount
# of data from the input set with replacement. So each network gets the same amount of
# training data but the sets are all different and may include some samples twice or more.
for i in range(args.num_bootstrap):
    print(f"Net {i + 1}")
    # tf.keras.backend.clear_session()
    adam = Adam(learning_rate=initial_lr, decay=0.00, amsgrad=args.amsgrad)
    temp_net = QuasarNET(input_shape=X[0, :, None].shape,
                          boxes=args.boxes,
                          nlines=len(args.lines) + len(args.lines_bal),
                          offset_activation_function=offset_activation)

    # Get the sample of spectra to train on.
    # This picks len(inds) number of items out of inds (with replacement)
    sample = rng.choice(inds, size=len(inds))

    box_sample = []
    sw_sample = []
    # print(box)
    # For some reason I picked e as the variable at first? what?
    for j, b in enumerate(box):
        box_sample.append(b[sample])
        sw_sample.append(sample_weight[j][sample])

    temp_net.compile(optimizer=adam, loss=loss, metrics=[])
    
    if i == 0:
        trainable_count = count_params(temp_net.trainable_weights)
        print("Number of Trainable Parameters:", trainable_count)
    
    temp_net.fit(X[sample][:, :, None], box_sample, epochs=args.epochs,
                batch_size=256, sample_weight=sw_sample,
                shuffle=True, verbose=2)


    # Save the weights so in we can reproduce the ensemble for running later
    # temp_net.save(weights_loc / f"net_{i}.h5", overwrite=True)
    tf.keras.models.save_model(temp_net, weights_loc / f"net_{i}.h5", include_optimizer=False)

    del temp_net

