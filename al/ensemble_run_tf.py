#!/usr/bin/env python3

""""
Uses an ensemble to generate confidence values and classifications for DESI data.

This script is hardcoded to load Guadalupe data.

This script will load the ensemble from the location provided by `--ensemble_loc` which
should be the same directory passed to `train_qnet_ensemble.py` in that script's
`--out-dir` argument. This script has its own `--out-dir` which defines the location
to save the csv file with the ensemble's results. 

QuasarNP is used for all model execution, rather than QuasarNET, to save on 
overhead introduced by Tensorflow and to enable execution on CPU only machines.

"""

import argparse
from datetime import date
import errno
import glob
import logging
import os
from pathlib import Path
import random
import time

from quasarnp.io import load_desi_coadd
from quasarnp.utils import process_preds

from tensorflow.keras.models import load_model
from quasarnet.models import QuasarNET, custom_loss

# For saving the results to a csv table
from astropy.io import ascii
from astropy.table import Table, vstack, unique

from somviz.som import SelfOrganizingMap, Grid

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import fitsio

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", type=str, required=True, help="where to save the output csv")
parser.add_argument("-i", "--ensemble_loc", type=str, required=True, help="where the ensemble weights are stored")
parser.add_argument("-d", "--data", type=str, required=True, help="base location of spectra to run through the ensemble")
parser.add_argument("-v", "--verbose", required=False, action="store_true")
parser.add_argument("-rt", "--random_tiles", required=False, action="store_true", 
                    help="whether or not to run on tiles in a random order rather than numeric.")
args = parser.parse_args()

# Line arguments for processing predictions.
default_lines = ["LYA", "CIV(1548)", "CIII(1909)", "MgII(2796)", "Hbeta", "Halpha"]
default_bal = ["CIV(1548)"]

# Threshold for "high-z quasar" classification
z_thresh = 2.1

# Confidence threshold for quasar classification.
c_thresh = 0.95
n_thresh = 1

if not Path("../loss.npy").is_file():
    raise FileNotFoundError(
    errno.ENOENT, os.strerror(errno.ENOENT), "loss.npy")

# Load the SOM
t1 = time.time()
som = SelfOrganizingMap(Grid(-45, -45))
som.fit(np.arange(5), maxiter=200, save="../", eta=0.001, use_saved=True)
t2 = time.time()
print(f"Took {t2-t1} seconds ({(t2-t1) / 60} minutes) to load SOM.")


out_loc = Path(args.out_dir)
if not os.path.isdir(out_loc):
    os.mkdir(out_loc)
    
today = date.today().strftime("%d_%m_%Y")
logging.basicConfig(filename=out_loc / f"{today}.log", level=logging.INFO)
logging.info(f"Took {t2-t1} seconds ({(t2-t1) / 60} minutes) to load SOM.")
    
ensemble_loc = Path(args.ensemble_loc)
ensemble = []
# Loop over the ensemble location and load all the weights files as quasarnp objects.
# j = 0
for weights_file in ensemble_loc.iterdir():
    ensemble.append(load_model(weights_file))
    # j += 1
    # if j > 6: break

# Tiles to run over.
guadalupe_loc = Path(args.data)    

table_name = "qnet_ensemble_guadalupe.csv"
table_loc = out_loc / table_name

# Creating an empty table just to allocate the variable.
out_table = Table(names=("TARGETID", "TILEID", "EXPID", "C_NOT", "C_QSO_LZ", "C_QSO_HZ", "BMU"),
                  dtype=(int, str, str, float, float, float, int))

# For skipping things we've already run
completed_exposures = []

# First will decide whether or not the very first tile run will determine the table struct
first = True
if Path(table_loc).is_file():
    out_table = Table.read(table_loc)
    first = False
    
    # note to self: these are saved as ints
    completed_exposures = np.unique(np.asarray(out_table["EXPID"][:]))

print(f"Completed exposures: {completed_exposures}")
logging.info(f"Completed exposures: {completed_exposures}")
j = 0 

tiles_to_run = sorted(guadalupe_loc.iterdir())
if args.random_tiles: random.shuffle(tiles_to_run)

for tile_dir in tiles_to_run:
    tile = tile_dir.name

    for p in tile_dir.iterdir():
        expid = p.name
        
        if int(expid) in completed_exposures: continue
        
        print("Loading...", tile, expid)
        logging.info(f"Loading... {tile} {expid}")
        
        # Resetting the categories per exposure
        cats = []
        X_datas = []
        tid_datas = []
        for f in sorted(p.glob(f"coadd-*{p.name}.fits")):
            X_temp, w = load_desi_coadd(f)
            X_datas.append(X_temp)
            
            with fitsio.FITS(f) as h:
                tids = structured_to_unstructured(h["FIBERMAP"].read(columns=["TARGETID"]))[w]
                tid_datas.append(tids)
            
        X_d = np.concatenate(X_datas)
        tids = np.concatenate(tid_datas)
        tids = tids.reshape(-1)
        
        t1 = time.time()
        print("Running...", tile, expid)
        logging.info(f"Running... {tile} {expid}")
        
        for i, temp_net in enumerate(ensemble):
            if i % 25 == 0: 
                print(f"Net {i}")
                logging.info(f"Net {i}")
            # Line predictions from this network
            p = temp_net.predict(X_d[:, :, None])
            c_line, z_line, zbest, c_line_bal, z_line_bal = process_preds(p, default_lines, default_bal, verbose=False)

            is_qso = np.sum(c_line > c_thresh, axis=0) >= n_thresh
            high_z = (zbest >= z_thresh) & is_qso

            # Non qsos stay at 0, low_z gets a 1, and high_z is a 2.
            label = is_qso.astype(int) + high_z.astype(int)

            cats.append(label)
          
        t2 = time.time()
        # This block computes the confidences from the categorization numbers
        cats = np.asarray(cats).T

        # Counts up how many times each of the 3 labels appears across all networks
        # and saves them per image. Divides by number of networks to get a "confidence"
        counts = np.zeros((cats.shape[0], 3))
        for i in range(0, 3):
            counts[:, i] = np.sum(cats == i, axis=1) 
        confs = counts / len(ensemble)
        
        print(f"Running spectra finished in {t2-t1} seconds ({(t2-t1) / 60} minutes), finding bmu...")
        logging.info(f"Running spectra finished in {t2-t1} seconds ({(t2-t1) / 60} minutes), finding bmu...")
        
        t1 = time.time()
        bmu = som.find_bmu(X_d)
        t2 = time.time()
        print(f"Took {t2-t1} seconds ({(t2-t1) / 60} minutes) to find BMU")
        logging.info(f"Took {t2-t1} seconds ({(t2-t1) / 60} minutes) to find BMU")
        
        # All that's left now to do is to add this to the running table of data and save it
        # We save after every exposure so if we get interrupted we've saved everything up to this point
        table_dict = {"TARGETID": tids, "TILEID": [int(tile)]*len(tids), "EXPID": [int(expid)]*len(tids),
                      "C_NOT": confs[:, 0], "C_QSO_LZ": confs[:, 1], "C_QSO_HZ": confs[:, 2],
                     "BMU": bmu}
        
        # This should work, right?
        if first:
            out_table = Table(table_dict)
            first = False
        else:
            out_table = vstack([out_table, Table(table_dict)])
            
        # Save the table and we're done with this exposure.
        ascii.write(out_table, table_loc, overwrite=True, format="csv")
        print(f"Finished {tile} {expid} and saved table.")
        logging.info(f"Finished {tile} {expid} and saved table.")
        
        
    j += 1
    
    # if j > 2:
    #     break
            
print("done")
logging.info("Finished all tiles and exposures!")
