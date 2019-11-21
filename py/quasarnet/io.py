from __future__ import print_function

from os.path import dirname

import numpy as np
from numpy import random
import fitsio
from random import randint
from os.path import dirname
from quasarnet import utils


def read_sdrq(sdrq):
    '''
    Constructs two dictionaries thing_id: class, z_conf, z,
            and (plate, mjd, fiberid): thing_id
    input: (str) full path to Superset_DRQ.fits
    output: (list of dictionaries)
    '''

    sdrq = fitsio.FITS(sdrq)
    thid = sdrq[1]["THING_ID"][:]
    class_person = sdrq[1]["CLASS_PERSON"][:]
    z_conf = sdrq[1]["Z_CONF_PERSON"][:]
    z_vi = sdrq[1]["Z_VI"][:]

    plate = sdrq[1]['PLATE'][:]
    mjd = sdrq[1]['MJD'][:]
    fiberid = sdrq[1]['FIBERID'][:]

    sdrq = {t:(c,zc,z) for t,c,zc,z in zip(thid, class_person, z_conf, z_vi)}
    pmf2tid = {(p,m,f):t for p,m,f,t in zip(plate, mjd, fiberid, thid)}

    return sdrq, pmf2tid

def read_truth(fi):
    '''
    reads a list of truth files and returns a truth dictionary

    Arguments:
        fi -- list of truth files (list of string)

    Returns:
        truth -- dictionary of THING_ID: metadata instance

    '''

    class metadata:
        pass

    cols = ['Z_VI','PLATE',
            'MJD','FIBERID','CLASS_PERSON',
            'Z_CONF_PERSON','BAL_FLAG_VI','BI_CIV']

    truth = {}

    for f in fi:
        h = fitsio.FITS(f)
        tids = h[1]['THING_ID'][:]
        cols_dict = {c.lower():h[1][c][:] for c in cols}
        h.close()
        for i,t in enumerate(tids):
            m = metadata()
            for c in cols_dict:
                setattr(m,c,cols_dict[c][i])
            truth[t] = m

    return truth

def read_data(fi, truth=None, z_lim=2.1, return_pmf=False, nspec=None):
    '''
    reads data from input file

    Arguments:
        fi -- list of data files (string iterable)
        truth -- dictionary thind_id => metadata
        z_lim -- hiz/loz cut (float)
        return_pmf -- if True also return plate,mjd,fiberid
        nspec -- read this many spectra

    Returns:
        tids -- list of thing_ids
        X -- spectra reformatted to be fed to the network (numpy array)
        Y -- truth vector (nqso, 5):
                           STAR = (1,0,0,0,0), GAL = (0,1,0,0,0)
                           QSO_LZ = (0,0,1,0,0), QSO_HZ = (0,0,0,1,0)
                           BAD = (0,0,0,0,1)
        z -- redshift (numpy array)
        bal -- 1 if bal, 0 if not (numpy array)
    '''

    tids = []
    X = []
    Y = []
    z = []
    bal = []

    if return_pmf:
        plate = []
        mjd = []
        fid = []

    for f in fi:
        print('INFO: reading data from {}'.format(f))
        h=fitsio.FITS(f)
        if nspec is None:
            nspec = h[1].get_nrows()
        aux_tids = h[1]['TARGETID'][:nspec].astype(int)
        print("INFO: found {} spectra in file {}".format(aux_tids.shape[0], f))

        ## remove thing_id == -1 or not in sdrq
        w = (aux_tids != -1)
        print("INFO: removing {} spectra with thing_id=-1".format((~w).sum()),flush=True)
        aux_tids = aux_tids[w]
        aux_X = h[0][:nspec,:]
        aux_X = aux_X[w]

        if truth is not None:
            w_in_truth = np.in1d(aux_tids, list(truth.keys()))
            print("INFO: removing {} spectra missing in truth".format((~w_in_truth).sum()),flush=True)
            aux_tids = aux_tids[w_in_truth]
            aux_X = aux_X[w_in_truth]

        if return_pmf:
            aux_plate = h[1]['PLATE'][:][w]
            aux_mjd = h[1]['MJD'][:][w]
            aux_fid = h[1]['FIBERID'][:][w]
            plate += list(aux_plate)
            mjd += list(aux_mjd)
            fid += list(aux_fid)

        X.append(aux_X)
        tids.append(aux_tids)

    tids = np.concatenate(tids)
    X = np.concatenate(X)

    if return_pmf:
        plate = np.array(plate)
        mjd = np.array(mjd)
        fid = np.array(fid)

    we = X[:,443:]
    w = we.sum(axis=1)==0
    print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    if return_pmf:
        plate = plate[~w]
        mjd = mjd[~w]
        fid = fid[~w]

    mdata = np.average(X[:,:443], weights = X[:,443:], axis=1)
    sdata = np.average((X[:,:443]-mdata[:,None])**2,
            weights = X[:,443:], axis=1)
    sdata=np.sqrt(sdata)

    w = sdata == 0
    print("INFO: removing {} spectra with zero flux".format(w.sum()))
    X = X[~w]
    tids = tids[~w]
    mdata = mdata[~w]
    sdata = sdata[~w]

    if return_pmf:
        plate = plate[~w]
        mjd = mjd[~w]
        fid = fid[~w]

    X = X[:,:443]-mdata[:,None]
    X /= sdata[:,None]

    if truth==None:
        if return_pmf:
            return tids,X,plate,mjd,fid
        else:
            return tids,X

    ## remove zconf == 0 (not inspected)
    observed = [(truth[t].class_person>0) or (truth[t].z_conf_person>0) for t in tids]
    observed = np.array(observed, dtype=bool)
    print("INFO: removing {} spectra that were not inspected".format((~np.array(observed)).sum()))
    tids = tids[observed]
    X = X[observed]

    if return_pmf:
        plate = plate[observed]
        mjd = mjd[observed]
        fid = fid[observed]

    ## fill redshifts
    z = np.zeros(X.shape[0])
    z[:] = [truth[t].z_vi for t in tids]

    ## fill bal
    bal = np.zeros(X.shape[0])
    bal[:] = [(truth[t].bal_flag_vi*(truth[t].bi_civ>0))-\
            (not truth[t].bal_flag_vi)*(truth[t].bi_civ==0) for t in tids]

    ## fill classes
    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    sdrq_class = np.array([truth[t].class_person for t in tids])
    z_conf = np.array([truth[t].z_conf_person for t in tids])

    Y = np.zeros((X.shape[0],nclasses))
    ## STAR
    w = (sdrq_class==1) & (z_conf==3)
    Y[w,0] = 1

    ## GALAXY
    w = (sdrq_class==4) & (z_conf==3)
    Y[w,1] = 1

    ## QSO_LZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z<z_lim) & (z_conf==3)
    Y[w,2] = 1

    ## QSO_HZ
    w = ((sdrq_class==3) | (sdrq_class==30)) & (z>=z_lim) & (z_conf==3)
    Y[w,3] = 1

    ## BAD
    w = z_conf != 3
    Y[w,4] = 1

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    print("INFO: {} spectra in returned dataset".format(tids.shape[0]))

    if return_pmf:
        return tids,X,Y,z,bal,plate,mjd,fid

    return tids,X,Y,z,bal

# TODO: move this somewhere else? Feels an odd place to keep it?
# Made a class utils.Wave to replace this. Has these llmin/llmax/dll as
# default but can add functionality to overwrite this with relative ease. Left
# without for now to avoid any potential issues.
llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3
nbins = int((llmax-llmin)/dll)
wave = 10**(llmin + np.arange(nbins)*dll)
nmasked_max = len(wave)+1

## spcframe = individual exposures of spectra
def read_spcframe(b_spcframe,r_spcframe):

    '''
    reads data from spcframes

    Arguments:
        b_spcframe -- spcframe from b part of spectrograph
        r_spcframe -- spcframe from r part of spectrograph

    Returns:
        fids -- fiberids ?
        data -- ?
    '''

    data = []
    fids = []

    hb = fitsio.FITS(b_spcframe)
    hr = fitsio.FITS(r_spcframe)
    target_bits = hb[5]["BOSS_TARGET1"][:]
    wqso = np.zeros(len(target_bits),dtype=bool)
    mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
    for i in mask:
        wqso = wqso | (target_bits & 2**i)
    ## SEQUELS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET0"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass

    ## EBOSS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET1"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass
    wqso = wqso>0
    print("INFO: found {} quasars in file {}".format(wqso.sum(),b_spcframe))

    plate = hb[0].read_header()["PLATEID"]
    fid = hb[5]["FIBERID"][:]
    fl = np.hstack((hb[0].read(),hr[0].read()))
    iv = np.hstack((hb[1].read()*(hb[2].read()==0),hr[1].read()*(hr[2].read()==0)))
    ll = np.hstack((hb[3].read(),hr[3].read()))

    fid = fid[wqso]
    fl = fl[wqso,:]
    iv = iv[wqso,:]
    ll = ll[wqso,:]

    for i in range(fl.shape[0]):
        fl_aux = np.zeros(nbins)
        iv_aux = np.zeros(nbins)
        bins = ((ll[i]-llmin)/dll).astype(int)
        wbin = (bins>=0) & (bins<nbins) & (iv[i]>0)
        bins=bins[wbin]
        c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
        fl_aux[:len(c)]=+c
        c = np.bincount(bins,weights=iv[i,wbin])
        iv_aux[:len(c)]=+c
        nmasked = (iv_aux==0).sum()
        if nmasked >= nmasked_max :
            print("INFO: skipping specrum {} with too many masked pixels {}".format(fid[i],nmasked))
            continue
        data.append(np.hstack((fl_aux,iv_aux)))
        fids.append(fid[i])

        assert ~np.isnan(fl_aux,iv_aux).any()

    if len(data)==0:
        return

    data = np.vstack(data)
    assert ~np.isnan(data).any()
    ## now normalize coadded fluxes
    norm = data[:,nbins:]*1.
    w = norm==0
    norm[w] = 1.
    data[:,:nbins]/=norm

    assert ~np.isnan(data).any()

    return fids, data

## ??
def read_spall(spall):

    '''
    reads data from spall file

    Arguments:
        spall -- filename of spall file

    Returns:
        tid -- array of tids
        pmf2tid -- dictionary mapping (plate,mjd,fiber_id) => thing_id
    '''

    ## Open the file, and read plate, mjd, fiberid, thing_id, specprimary.
    spall = fitsio.FITS(spall)
    plate = spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fid = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:].astype(int)
    specprim = spall[1]["SPECPRIMARY"][:]

    ## Construct a dictionary mapping plate, mjd and fiberid to thing_id.
    pmf2tid = {(p,m,f):t for p,m,f,t,s in zip(plate,mjd,fid,tid,specprim)}
    spall.close()

    return tid, pmf2tid

## spplate = coadded spectra
def read_spplate(fin, fibers):

    '''
    reads data from spplates

    Arguments:
        fin -- spplate file to read
        fibers -- list of fiberids

    Returns:
        fids -- fiberids ?
        fl -- flux/iv ?
    '''

    ## Open the file and read data from the header.
    h = fitsio.FITS(fin)
    head = h[0].read_header()
    c0 = head["COEFF0"]
    c1 = head["COEFF1"]
    p = head["PLATEID"]
    m = head["MJD"]

    ## Filter the fiberids in the file by those we're interested in.
    fids = h[5]["FIBERID"][:]
    wqso = np.in1d(fids, fibers)
    fids = fids[wqso]

    ## Construct the grids for flux and iv.
    nspec = len(fibers)
    wave_out = utils.Wave()
    fl = np.zeros((nspec, wave_out.nbins))
    iv = np.zeros((nspec, wave_out.nbins))

    ## Read the data from file.
    fl_aux = h[0].read()[wqso,:]
    iv_aux = h[1].read()[wqso,:]*((h[2].read()[wqso]&2**25)==0)

    ## Calculate how to rebin the data.
    wave_grid = 10**(c0 + c1*np.arange(fl_aux.shape[1]))
    bins, w = rebin_wave(wave_grid,wave_out)
    bins = bins[w]
    fl_aux = fl_aux[:,w]
    iv_aux =iv_aux[:,w]

    ## For each spectrum, rebin the flux and iv and add them to the pre-
    ## constructed grids.
    for i in range(nspec):
        c = np.bincount(bins, weights=fl_aux[i]*iv_aux[i])
        fl[i,:len(c)] += c
        c = np.bincount(bins, weights = iv_aux[i])
        iv[i,:len(c)]+=c

    ## Normalise the flux and stack fl and iv.
    w = iv>0
    fl[w] /= iv[w]
    fl = np.hstack((fl,iv))

    ## Filter out spectra with too many bad pixels.
    wbad = (iv==0)
    w = (wbad.sum(axis=1)>nmasked_max)
    print('INFO: rejecting {} spectra with too many bad pixels'.format(w.sum()))
    if (~w).sum()==0:
        return None
    fl = fl[~w,:]
    fids = fids[~w]

    return fids, fl

## ??
def read_exposures(plates,pmf2tid,nplates=None, random_exp=False):
    '''
    Given a list of plates, returns the thing_id list and the
        rebinned fluxes for all the exposures in the plates

    input:
        -- plates: list of str. List of paths to the spPlate files
        -- pmf2tid: dictionary containing (plate, mjd, fiber): thing_id
        -- nplates: use only the first nplates in the list
        -- random_exp: read only one random exposure from all the available
            exposures
    output: thid, data
        -- thid: list of thing ids of length equal the the number of exposures
        -- data: numpy array of float of shape (nexps, nbins)
    '''

    data = []
    read_plates = 0
    tids = []

    plate_mjd_in_pmf2tid = np.empty(len(pmf2tid), dtype=object)
    print('calculating plates-mjd combos')
    plate_mjd_in_pmf2tid[:] =[(k[0], k[1]) for k in pmf2tid.keys()]
    print('uniq-ing')
    plate_mjd_in_pmf2tid = list(np.unique(plate_mjd_in_pmf2tid))
    print('done')

    if nplates is not None:
        plates = plates[:nplates]
    for p in plates:
        h=fitsio.FITS(p)
        head = h[0].read_header()
        plateid = head['PLATEID']
        m = head['MJD']
        if (plateid,m) not in plate_mjd_in_pmf2tid:
            print('{} {} not in list'.format(plateid,m))
            continue

        exps = []
        ## read b,r exposures
        try:
            nexp_b = head["NEXP_B1"]+head["NEXP_B2"]
        except:
            continue
        if nexp_b>99:
            nexp_b=99
        for exp in range(nexp_b):
            str_exp = str(exp+1)
            if exp<9:
                str_exp = '0'+str_exp
            exp_b = head["EXPID{}".format(str_exp)][:11]
            exp_r = exp_b.replace("b", "r")
            exps.append((exp_b, exp_r))

        exps_spectro_1 = [e for e in exps if 'b1' in e[0]]
        exps_spectro_2 = [e for e in exps if 'b2' in e[0]]
        if random_exp:
            irand1 = randint(0,len(exps_spectro_1)-1)
            irand2 = randint(0,len(exps_spectro_2)-1)
            exps = [exps_spectro_1[irand1], exps_spectro_2[irand2]]

        for exp_b, exp_r in exps:
            spcframe_b = dirname(p)+"/spCFrame-{}.fits".format(exp_b)
            spcframe_r = dirname(p)+"/spCFrame-{}.fits".format(exp_r)
            res = read_spcframe(spcframe_b, spcframe_r)
            if res is not None:
                plate_fid, plate_data = res
                data.append(plate_data)
                tids = tids + [pmf2tid[(plateid,m,f)] for f in plate_fid]

        if nplates is not None:
            if len(data)//2==nplates:
                break

    data = np.vstack(data)

    return tids, data

# TODO: what does this do exactly? Search for uses.
# Doesn't seem to be used in the repo at all.
# Maybe in scripts? Haven't seen any exts with 'DATA' as a name...
def export_data(fout,tids,data):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(data,extname="DATA")
    tids = np.array(tids)
    h.write([tids],names=["TARGETID"],extname="METADATA")
    h.close()

# TODO: should this go in utils maybe?
from .utils import absorber_IGM
from scipy.interpolate import interp1d
def box_offset(z, line='LYA', nboxes = 13):

    wave = utils.Wave()

    ## Interpolate the locations of the line in each object in terms of
    ## wavelength to the position in terms of the number of boxes.
    wave_to_i = interp1d(wave.wave_grid, np.arange(len(wave.wave_grid)),
            bounds_error=False, fill_value=-1)
    wave_line = (1+z)*absorber_IGM[line]
    pos = wave_to_i(wave_line)/len(wave.wave_grid)*nboxes
    ipos = np.floor(pos).astype(int)

    box = np.zeros((len(z), nboxes))
    offset = np.zeros((len(z), nboxes))

    w = (ipos>=0)
    box[w, ipos[w]] = 1
    offset[w, ipos[w]] = (pos-ipos)[w]
    weights = np.ones(len(z))
    weights[~w]=0

    return box, offset, weights

# TODO: is this the right place for this?
def objective(z, Y, bal, lines=['LYA'], lines_bal=['CIV(1548)'], nboxes=13):

    box=[]
    sample_weight = []
    for l in lines:
        # TODO: Do we need to use "weight_line"?
        box_line, offset_line, weight_line = box_offset(z,
                line = l, nboxes=nboxes)

        w = (Y.argmax(axis=1)==2) | (Y.argmax(axis=1)==3)
        ## set to zero where object is not a QSO
        ## (the line confidence should be zero)
        box_line[~w]=0
        box.append(np.concatenate([box_line, offset_line], axis=-1))
        sample_weight.append(np.ones(Y.shape[0]))

    for l in lines_bal:
        box_line, offset_line, weight_line = box_offset(z,
                line = l, nboxes=nboxes)

        ## set to zero for non-quasars
        wqso = (Y.argmax(axis=1)==2) | (Y.argmax(axis=1)==3)
        box_line[~wqso] = 0

        ## set to zero for confident non-bals:
        wnobal = (bal==-1)
        box_line[wnobal] = 0

        ## use only spectra where visual flag and bi_civ do agree
        bal_weight = bal != 0
        box.append(np.concatenate([box_line, offset_line], axis=-1))
        sample_weight.append(bal_weight)

    return box, sample_weight





################################################################################
## DESI RELATED THINGS


## WIP
def read_desi_sim_truth(fi,f_qsotemp=None,f_baltemp=None):
    '''
    reads a list of desi truth files and returns a truth dictionary

    Arguments:
        fi -- list of truth files (list of string)

    Returns:
        truth -- dictionary of THING_ID: metadata instance

    '''

    class metadata:
        pass

    spectypes = ['BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'WD']

    ## These are not needed, but give an idea of the information loaded.
    # Data in the 'TRUTH' hdu.
    #cols_target = ['TARGETID', 'MOCKID', 'TRUEZ', 'TRUESPECTYPE', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TEMPLATEID', 'SEED', 'MAG', 'MAGFILTER', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']
    # Data in the 'TRUTH_QSO' hdu.
    #cols_qso = ['TARGETID', 'MABS_1450', 'SLOPES', 'EMLINES', 'BAL_TEMPLATEID', 'TRUEZ_NORSD']

    ## Cycle through the files, loading data from each one.
    truth = {}
    for f in fi:
        # Load data from the 'TRUTH' HDU which contains general target data.
        h = fitsio.FITS(f)
        hdu = h['TRUTH']
        tids = hdu['TARGETID'][:]
        cols_dict = {c.lower(): hdu[c][:] for c in hdu.get_colnames()}
        h.close()
        for i,t in enumerate(tids):
            m = metadata()
            for c in cols_dict:
                setattr(m,c,cols_dict[c][i])
            truth[t] = m

        # For each object type, also add data from the specific HDUs.
        for st in spectypes:
            hdu = h['TRUTH_{}'.format(st)]
            tids = hdu['TARGETID'][:]
            cols_dict = {c.lower(): hdu[c][:] for c in hdu.get_colnames()}
            for i,t in enumerate(tids):
                m = truth[t]
                for c in cols_dict:
                    setattr(m,c,cols_dict[c][i])

    return truth

## WIP
def read_desi_data(fi, truth=None, z_lim=2.1, return_pmf=False, nspec=None):
    '''
    reads data from input file

    Arguments:
        fi -- list of data files (string iterable)
        truth -- dictionary targetid => metadata
        z_lim -- hiz/loz cut (float)
        nspec -- read this many spectra

        # TODO: plate and mjd are deprecated? Replace with something?
        # Option removed for the moment.
        return_pmf -- if True also return plate,mjd,fiberid

    Returns:
        tids -- list of thing_ids
        X -- spectra reformatted to be fed to the network (numpy array)
        Y -- truth vector (nqso, 5):
                           STAR (STAR or WD) =      (1,0,0,0,0),
                           GAL (BGS, LRG or ELG) =  (0,1,0,0,0),
                           QSO_LZ =                 (0,0,1,0,0),
                           QSO_HZ =                 (0,0,0,1,0),
                           BAD =                    (0,0,0,0,1)
        z -- redshift (numpy array)
        bal -- 1 if bal, 0 if not (numpy array)
    '''

    tids = []
    utids_list = []
    fl_list = []
    iv_list = []
    Y = []
    z = []
    bal = []

    if return_pmf:
        plate = []
        mjd = []
        fid = []

    wave_out = utils.Wave()

    ## Go through the files, loading data from each one.
    for f in fi:
        h = fitsio.FITS(f)

        # Apply the mask.
        wqso = h[1]['DESI_TARGET'][:]
        nqso_f = wqso.sum()
        print("INFO: found {} quasar targets".format(nqso_f))
        tids = h[1]["TARGETID"][:][wqso]
        utids = np.unique(tids)

        nspec = len(utids)
        fl = np.zeros((nspec, nbins))
        iv = np.zeros((nspec, nbins))
        if nspec == 0: return None
        for band in ["B", "R", "Z"]:
            h_wave = h["{}_WAVELENGTH".format(band)].read()

            ## new system
            bins, w = rebin_wave(h_wave,wave_out)
            h_wave = h_wave[w]
            bins = bins[w]

            # Filter by valid pixels and QSOs.
            fl_aux = h["{}_FLUX".format(band)].read()[:,w]
            iv_aux = h["{}_IVAR".format(band)].read()[:,w]
            fl_aux = fl_aux[wqso]
            iv_aux = iv_aux[wqso]

            ivfl_aux = fl_aux*iv_aux
            for i,t in enumerate(tids):
                j = np.argwhere(utids==t)[0]
                c = np.bincount(bins, weights = ivfl_aux[i])
                fl[j,:len(c)] += c
                c = np.bincount(bins, weights = iv_aux[i])
                iv[j,:len(c)] += c

        # Normalise flux by dividing by summed ivars.
        w = iv>0
        fl[w] /= iv[w]

        # Check that there are no overlaps between the objects in this file and
        # in previous ones.
        if f != fin[0]:
            check = np.in1d(utids,global_utids)
            if check.sum() > 0:
                print('INFO: the following thing_ids are found in multiple files:')
                print(utids[check])
                utids = utids[check]
                fl = fl[check,:]
                iv = iv[check,:]
                nqso_f = check.sum()

        # Add the flux and iv arrays for this file to a list.
        fl_list += [fl]
        iv_list += [iv]
        utids_list += [utids]
        nqso += nqso_f

        global_utids = np.concatenate(utids_list)

    # Concatenate the lists to arrays.
    fl = np.concatenate(fl_list,axis=0)
    iv = np.concatenate(iv_list,axis=0)
    utids = np.concatenate(utids_list)


    # TODO: Check what we're doing with spectra from same object
    # related to difference between tids and utids
    tids = utids
    X = np.hstack((fl,iv))

    ## If desired, restrict the number of spectra.
    if nspec is None:
        nspec = X.shape[0]
    tids = tids[:nspec].astype(int)
    X = X[:nspec,:]
    print("INFO: found {} spectra".format(tids.shape[0]))

    # TODO: is this the right thing to do?
    ## remove thing_id < 0 or not in sdrq
    w = (utids < 0)
    print("INFO: removing {} spectra with thing_id<0".format((~w).sum()),flush=True)
    tids = tids[w]
    X = X[w,:]

    if truth is not None:
        w_in_truth = np.in1d(tids, list(truth.keys()))
        w *= w_in_truth
        print("INFO: removing {} spectra missing in truth".format((~w_in_truth).sum()),flush=True)
        tids = tids[w_in_truth]
        X = X[w_in_truth]

    # TODO: this feels dangerous really - it should just be a check that our spectra are the right size?
    we = X[:,443:]

    w = we.sum(axis=1)==0
    print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    mdata = np.average(X[:,:443], weights = X[:,443:], axis=1)
    sdata = np.average((X[:,:443]-mdata[:,None])**2,
            weights = X[:,443:], axis=1)
    sdata = np.sqrt(sdata)

    w = (sdata == 0)
    print("INFO: removing {} spectra with zero flux".format(w.sum()))
    X = X[~w]
    tids = tids[~w]
    mdata = mdata[~w]
    sdata = sdata[~w]

    ## Enforce that the spectra each have mean 0 and sigma 1
    X = X[:,:443]-mdata[:,None]
    X /= sdata[:,None]

    if truth==None:
        return tids,X

    ## This isn't relevant yet. Will need to be implemented when looking at SV data.
    """
    ## Remove zconf == 0 (not inspected)
    observed = [(truth[t].class_person>0) or (truth[t].z_conf_person>0) for t in tids]
    observed = np.array(observed, dtype=bool)
    print("INFO: removing {} spectra that were not inspected".format((~np.array(observed)).sum()))
    tids = tids[observed]
    X = X[observed]
    """

    ## fill redshifts
    z = np.zeros(X.shape[0])
    try:
        z[:] = [truth[t].z_vi for t in tids]
        print('INFO: using attribute {} for true z'.format('z_vi'))
    except AttributeError:
        z[:] = [truth[t].TRUEZ for t in tids]
        print('INFO: using attribute {} for true z'.format('TRUEZ'))

    # TODO: Need to have option to use bal templates here?
    """
    ## fill bal
    bal = np.zeros(X.shape[0])
    bal[:] = [(truth[t].bal_flag_vi*(truth[t].bi_civ>0))-\
            (not truth[t].bal_flag_vi)*(truth[t].bi_civ==0) for t in tids]
    """

    ## fill classes
    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    truespectype = np.array([truth[t].TRUESPECTYPE for t in tids])
    truez = np.array([truth[t].TRUEZ for t in tids])

    Y = np.zeros((X.shape[0],nclasses))
    ## STAR
    w = ((truespectype=='STAR') | (truespectype=='WD'))
    Y[w,0] = 1

    ## GALAXY
    w = (truespectype=='GALAXY')
    Y[w,1] = 1

    ## QSO_LZ
    w = (truespectype=='QSO') & (truez<z_lim)
    Y[w,2] = 1

    ## QSO_HZ
    w = (truespectype=='QSO') & (truez>=z_lim)
    Y[w,3] = 1

    # TODO: don't think there are any bad spectra atm. Need to reintroduce this
    # once there are, and also introduce 'z_good' filters in the above.
    """
    ## BAD
    w = z_conf != 3
    Y[w,4] = 1
    """

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    print("INFO: {} spectra in returned dataset".format(tids.shape[0]))

    return tids,X,Y,z,bal

def rebin_wave(wave_grid_in,wave_out):

    # Potential new system.
    #wave_grid_out = wave_out.wave_grid
    #wave_edges = np.concatenate(([wave_grid_out[0]-(wave_grid_out[1]-wave_grid_out[0])/2],(wave_grid_out[1:]+wave_grid_out[:-1])/2,[wave_grid_out[-1]+(wave_grid_out[-1]-wave_grid_out[-2])/2]))
    #bins = np.searchsorted(wave_edges,wave_grid_in)-1

    # Old system:
    # This system treats the output wave grid as the lower bounds of the bins.
    # It is implemented consistently and so does not introduce a bias as a
    # result of the floor function.
    bins = np.floor((np.log10(wave_grid_in)-wave_out.llmin)/wave_out.dll).astype(int)
    w = (bins>=0) & (bins<wave_out.nbins)

    return bins, w

# TODO: Make sure that this is up to date.
# TODO: Integrate with function to read desi truth like 'read_data'
def read_desi_spectra(fin, ignore_quasar_mask=False, verbose=True):

    # Obtain the mask from desitarget if available.
    try:
        from desitarget import desi_mask
        quasar_mask = desi_mask.mask('QSO')
    except:
        print("WARN: can't load desi_mask, using hardcoded targetting value!")
        quasar_mask = 2**2

    if not isinstance(fin,list):
        fin = [fin]

    fl_list = []
    iv_list = []
    utids_list = []
    nqso = 0

    wave_out = utils.Wave()

    for f in fin:
        h = fitsio.FITS(f)

        # Apply the mask.
        wqso = h[1]['DESI_TARGET'][:] & quasar_mask
        if ignore_quasar_mask:
            wqso |= 1
        wqso = (wqso>0)
        nqso_f = wqso.sum()
        print("INFO: found {} quasar targets".format(nqso_f))
        tids = h[1]["TARGETID"][:][wqso]
        utids = np.unique(tids)

        nspec = len(utids)
        fl = np.zeros((nspec, nbins))
        iv = np.zeros((nspec, nbins))
        if nspec == 0: return None
        for band in ["B", "R", "Z"]:
            h_wave = h["{}_WAVELENGTH".format(band)].read()

            bins, w = rebin_wave(h_wave,wave_out)
            h_wave = h_wave[w]
            bins = bins[w]

            # Filter by valid pixels and QSOs.
            fl_aux = h["{}_FLUX".format(band)].read()[:,w]
            iv_aux = h["{}_IVAR".format(band)].read()[:,w]
            fl_aux = fl_aux[wqso]
            iv_aux = iv_aux[wqso]

            ivfl_aux = fl_aux*iv_aux
            for i,t in enumerate(tids):
                j = np.argwhere(utids==t)[0]
                c = np.bincount(bins, weights = ivfl_aux[i])
                fl[j,:len(c)] += c
                c = np.bincount(bins, weights = iv_aux[i])
                iv[j,:len(c)] += c

        # Normalise flux by dividing by summed ivars.
        w = iv>0
        fl[w] /= iv[w]

        # Check that there are no overlaps between the objects in this file and
        # in previous ones.
        if f != fin[0]:
            check = np.in1d(utids,global_utids)
            if check.sum() > 0:
                print('INFO: the following thing_ids are found in multiple files:')
                print(utids[check])
                utids = utids[check]
                fl = fl[check,:]
                iv = iv[check,:]
                nqso_f = check.sum()

        # Add the flux and iv arrays for this file to a list.
        fl_list += [fl]
        iv_list += [iv]
        utids_list += [utids]
        nqso += nqso_f

        global_utids = np.concatenate(utids_list)

    # Concatenate the lists to arrays.
    fl = np.concatenate(fl_list,axis=0)
    iv = np.concatenate(iv_list,axis=0)
    utids = np.concatenate(utids_list)
    fl = np.hstack((fl,iv))

    print("INFO: found {} good spectra".format(nqso))
    return utids, fl

# TODO: generalise this to allow for many files to be loaded at once.
def read_desi_truth(fin):
    h = fitsio.FITS(fin)
    truth = {}
    for t,c,z in zip(h[1]["TARGETID"][:], h[1]["TRUESPECTYPE"][:], h[1]["TRUEZ"][:]):
        c = c.strip()
        if c==b"QSO":
            c=3
        elif c==b"GALAXY":
            c=4
        elif c==b"STAR":
            c=1
        assert isinstance(c,int)
        truth[t] = (c,3,z)
    return truth
