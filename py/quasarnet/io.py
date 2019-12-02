from __future__ import print_function

from os.path import dirname

import numpy as np
from numpy import random
import fitsio
from random import randint
from os.path import dirname

from quasarnet import utils

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

################################################################################
## Read raw data to be parsed.

def read_sdrq(sdrq, mode='BOSS'):
    '''
    Reads a superset DRQ file, and constructs two dictionaries:
     - {thing_id: class, z_conf, z}
     - {(plate, mjd, fiberid): thing_id}
    input: (str) full path to Superset_DRQ.fits
    output: (list of dictionaries, columns and column names)
    '''

    tid_field = utils.get_tid_field(mode)
    spid_fields = utils.get_spectrum_id_fields(mode)
    truth_fields = utils.get_truth_fields(mode)

    sdrq = fitsio.FITS(sdrq)

    data = {}
    f_dicts = [tid_field, spid_fields, truth_fields]
    for f_dict in f_dicts:
        for k in f_dict.keys():
            data[k] = sdrq[1][f_dict[k]][:]

    ## For a simulation, we have an absolute truth, and so add "confidence"
    ## artificially.
    if mode == 'DESISIM':
        data['Z_CONF'] = 4*np.ones_like(data['Z'])

    sdrq.close()

    ## Construct dictionaries {targetid: class, z_conf, z}, and
    ## {(spid0, spid1, spid2): targetid}.
    t2t_data = zip(data['TARGETID'],data['OBJCLASS'],data['Z_CONF'],data['Z'])
    s2t_data = zip(data['SPID0'],data['SPID1'],data['SPID2'],data['TARGETID'])
    tid2truth = {t:(c,zc,z) for t,c,zc,z in t2t_data}
    spid2tid = {(s0,s1,s2):t for s0,s1,s2,t in s2t_data}

    cols = list(data.values())
    colnames = list(data.keys())

    return tid2truth, spid2tid, cols, colnames

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

## spall = metadata for all spectra
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

## Read the spectra files from BOSS (spplate) or DESI (desi_spectra)
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
    bins, w = utils.rebin_wave(wave_grid,wave_out)
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

def read_desi_spectra_list(fin, ignore_quasar_mask=False, verbose=True, targeting_bits='DESI_TARGET'):

    '''
    reads data from DESI spectra files (per HEALPix pixel)

    Arguments:
        fin -- list of spectra files to read
        ignore_quasar_mask -- include all spectra, not just quasar targets
        verbose -- chatty or not
        targeting_bits -- which targeting bits to use

    Returns:
        meta -- dictionary of metadata
        fl -- hstacked flux/iv array
    '''

    # Obtain the mask from desitarget if available.
    if ignore_quasar_mask:
        quasar_mask = -1
    else:
        quasar_mask = utils.get_quasar_mask(verbose=verbose)

    if not isinstance(fin,list):
        fin = [fin]

    tids_list = []
    spid0_list = []
    spid1_list = []
    spid2_list = []
    fl_list = []
    iv_list = []
    nspec = 0
    global_tids = []

    for i,f in enumerate(fin):

        aux = read_desi_spectra(f, quasar_mask, verbose=verbose, targeting_bits=targeting_bits)

        if aux:
            tids, spid0, spid1, spid2, fl, iv = aux

            # Check that there are no overlaps between the objects in this file and
            # in previous ones.
            if f != fin[0]:
                check = np.in1d(tids,global_tids)
                if check.sum() > 0:
                    if verbose:
                        print('INFO: the following thing_ids are found in multiple files:')
                        print(tids[check])
                    tids = tids[check]
                    spid0 = spid0[check]
                    spid1 = spid1[check]
                    spid2 = spid2[check]
                    fl = fl[check,:]
                    iv = iv[check,:]

            # Add the flux and iv arrays for this file to a list.
            tids_list += [tids]
            spid0_list += [spid0]
            spid1_list += [spid1]
            spid2_list += [spid2]
            fl_list += [fl]
            iv_list += [iv]
            nspec += len(tids)
            global_tids = np.concatenate(tids_list)

        else:
            if verbose:
                print('INFO: {} has no quasar spectra'.format(f))

        print('INFO: read {:4d}/{:4d} files ({:.2%})'.format(i+1,len(fin),(i+1)/len(fin)),end='\r')

    print('')

    if len(tids_list)>0:
        # Concatenate the lists to arrays.
        tids = np.concatenate(tids_list)
        spid0 = np.concatenate(spid0_list)
        spid1 = np.concatenate(spid1_list)
        spid2 = np.concatenate(spid2_list)
        fl = np.concatenate(fl_list,axis=0)
        iv = np.concatenate(iv_list,axis=0)
        fliv = np.hstack((fl,iv))
        if verbose:
            print("INFO: found {} good spectra".format(nspec))
        return tids, spid0, spid1, spid2, fliv

    else:
        print('WARN: no quasar spectra found in given file list.')
        return None

def read_desi_spectra(f, quasar_mask, verbose=True, targeting_bits='DESI_TARGET'):

    tid_field = utils.get_tid_field('DESI')
    spid_fields = utils.get_spectrum_id_fields('DESI')

    h = fitsio.FITS(f)

    wqso = ((h[1][targeting_bits][:] & quasar_mask)>0)

    nspec_init = wqso.sum()
    if nspec_init == 0: return None

    if verbose:
        print("INFO: found {} target spectra".format(nqso_f))

    tids = h[1][tid_field['TARGETID']][:][wqso]
    spid0 = h[1][spid_fields['SPID0']][:][wqso]
    spid1 = h[1][spid_fields['SPID1']][:][wqso]
    spid2 = h[1][spid_fields['SPID2']][:][wqso]

    ## Here, we determine what to do with spectra that duplicate in some way.
    met = [(t,s0,s1,s2) for t,s0,s1,s2 in zip(tids,spid0,spid1,spid2)]
    umet = [tuple(x) for x in np.unique(met,axis=0)]

    ## Remove any entries with duplicated metadata.
    w = np.zeros(tids.shape).astype('bool')
    for um in umet:
        j = np.where([um==M for M in met])[0]
        if len(j)>1:
            w[j[0]] = 1
            if verbose:
                print('WARN: (tid,spid)={} is duplicated'.format(met))
        else:
            w[j] = 1

    tids = tids[w]
    spid0 = spid0[w]
    spid1 = spid1[w]
    spid2 = spid2[w]

    nspec = len(tids)
    fl = np.zeros((nspec, nbins))
    iv = np.zeros((nspec, nbins))

    if nspec == 0: return None

    wave_out = utils.Wave()

    for band in ["B", "R", "Z"]:
        h_wave = h["{}_WAVELENGTH".format(band)].read()

        bins, w = utils.rebin_wave(h_wave,wave_out)
        h_wave = h_wave[w]
        bins = bins[w]

        # Filter by valid pixels and QSOs.
        fl_aux = h["{}_FLUX".format(band)].read()[:,w]
        iv_aux = h["{}_IVAR".format(band)].read()[:,w]
        fl_aux = fl_aux[wqso]
        iv_aux = iv_aux[wqso]
        ivfl_aux = fl_aux*iv_aux

        for i,t in enumerate(tids):
            c = np.bincount(bins, weights = ivfl_aux[i])
            fl[i,:len(c)] += c
            c = np.bincount(bins, weights = iv_aux[i])
            iv[i,:len(c)] += c

    # Normalise flux by dividing by summed ivars.
    w = iv>0
    fl[w] /= iv[w]

    return tids, spid0, spid1, spid2, fl, iv

# TODO: write this.
def read_bal_data_drq(drq, mode='BOSS'):

    return bal_flag, bi_civ

## Simulated DESI specific functions.
def read_bal_data_desisim(truth,bal_templates):
    """
    Use a truth file and a BAL templates file to construct a dictionary mapping
    targetid to (bal_flag,bi_civ).

    Inputs:
     - truth: filename of truth file, str
     - bal_templates: filename of BAL templates file, str
    Outputs:
     - bal_data: {tid: (bal_flag,bi_civ)}, dict
    """

    ## Open the truth file, extract the templateid corresponding to each
    ## targetid.
    h = fitsio.FITS(truth)
    tids = h['TRUTH_QSO']['TARGETID'][:]
    bal_templateid = h['TRUTH_QSO']['BAL_TEMPLATEID'][:]
    h.close()

    ## Open the templates file, extract the bi_civ for each templateid.
    h = fitsio.FITS(bal_templates)
    bi_civ_templates = h['METADATA']['BI_CIV'][:]
    h.close()

    ## Exclude templateids<0 (i.e. no template used)
    w = (bal_templateid>=0)
    bi_civ = bi_civ_templates[bal_templateid[w]]
    tids = tids[w]

    ## Construct the dictionary mapping targetid to (bal_flag,bi_civ).
    bal_data = {t:(1,bi) for t,bi in zip(tids,bi_civ)}

    return bal_data

def read_truth_desisim(truth):

    tid_field = utils.get_tid_field('DESISIM')
    truth_fields = utils.get_truth_fields('DESISIM')

    h = fitsio.FITS(truth)

    tr_dict = {}
    tr_dict[tid_field['TARGETID']] = h[1][tid_field['TARGETID']][:]
    for k in truth_fields.keys():
        tr_dict[k] = h[1][truth_fields[k]][:]

    h.close()

    return tr_dict

def read_targets_desisim(targets,targeting_bits):

    tid_field = utils.get_tid_field('DESISIM')

    h = fitsio.FITS(targets)

    ta_dict = {}
    ta_dict[tid_field['TARGETID']] = h[1][tid_field['TARGETID']][:]
    ta_dict[targeting_bits] = h[1][targeting_bits][:]

    h.close()

    return ta_dict

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

################################################################################
## Read data after parsing.

def read_truth(fi, mode='BOSS'):
    '''
    reads a list of truth files and returns a truth dictionary

    Arguments:
        fi -- list of truth files (list of string)

    Returns:
        truth -- dictionary of THING_ID: truth data instance

    '''

    class metadata:
        pass

    """
    # Removed this, instead just read the cols from each file
    cols = ['Z_VI','PLATE',
            'MJD','FIBERID','CLASS_PERSON',
            'Z_CONF_PERSON','BAL_FLAG_VI','BI_CIV']
    """

    tid_field = utils.get_tid_field(mode)
    truth_fields = utils.get_truth_fields(mode)
    bal_fields = utils.get_bal_fields(mode)

    truth = {}

    ## Cycle through the files, extracting data from each one.
    for f in fi:
        # Open the file and get the tids.
        h = fitsio.FITS(f)
        tids = h[1][tid_field['TARGETID']][:]
        # Cycle through each tid.
        for i,t in enumerate(tids):
            m = metadata()
            # For each of the important field groups:
            for fd in [truth_fields,bal_fields]:
                # For each key:
                for k in fd.keys():
                    # Get the data from the column corresponding to that key's
                    # corresponding value, and add it to the metadata.
                    setattr(m,k,h[1][k][i])
            truth[t] = m
        h.close()

    return truth

def read_data(fi, truth=None, z_lim=2.1, return_spid=False, nspec=None, mode='BOSS'):
    '''
    reads data from input file

    Arguments:
        fi -- list of data files (string iterable)
        truth -- dictionary thind_id => metadata
        z_lim -- hiz/loz cut (float)
        return_spid -- if True also return tuple spectrum identifier
        nspec -- read this many spectra
        mode -- which data format are we using

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

    if return_spid:
        spid0 = []
        spid1 = []
        spid2 = []

    for f in fi:
        print('INFO: reading data from {}'.format(f))
        h = fitsio.FITS(f)
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

        if return_spid:
            try:
                aux_spid0 = h[1]['SPID0'][:][w]
                aux_spid1 = h[1]['SPID1'][:][w]
                aux_spid2 = h[1]['SPID2'][:][w]
            except ValueError:
                aux_spid0 = h[1]['PLATE'][:][w]
                aux_spid1 = h[1]['MJD'][:][w]
                aux_spid2 = h[1]['FIBERID'][:][w]
            spid0 += list(aux_spid0)
            spid1 += list(aux_spid1)
            spid2 += list(aux_spid2)

        X.append(aux_X)
        tids.append(aux_tids)

    tids = np.concatenate(tids)
    X = np.concatenate(X)

    if return_spid:
        spid0 = np.array(spid0)
        spid1 = np.array(spid1)
        spid2 = np.array(spid2)

    we = X[:,443:]
    w = we.sum(axis=1)==0
    print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    if return_spid:
        spid0 = spid0[~w]
        spid1 = spid1[~w]
        spid2 = spid2[~w]

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

    if return_spid:
        spid0 = spid0[~w]
        spid1 = spid1[~w]
        spid2 = spid2[~w]

    X = X[:,:443]-mdata[:,None]
    X /= sdata[:,None]

    if truth==None:
        if return_spid:
            return tids,X,spid0,spid1,spid2
        else:
            return tids,X

    ## remove zconf == 0 (not inspected)
    observed = [(truth[t].objclass>0) or (truth[t].z_conf>0) for t in tids]
    observed = np.array(observed, dtype=bool)
    print("INFO: removing {} spectra that were not inspected".format((~np.array(observed)).sum()))
    tids = tids[observed]
    X = X[observed]

    if return_spid:
        spid0 = spid0[observed]
        spid1 = spid1[observed]
        spid2 = spid2[observed]

    ## fill redshifts
    z = np.zeros(X.shape[0])
    z[:] = [truth[t].z for t in tids]

    ## fill bal
    bal = np.zeros(X.shape[0])
    bal[:] = [(truth[t].bal_flag*(truth[t].bi_civ>0))-\
            (not truth[t].bal_flag)*(truth[t].bi_civ==0) for t in tids]

    ## fill classes
    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    nclasses = 5
    objclass = np.array([truth[t].objclass for t in tids])
    z_conf = np.array([truth[t].z_conf for t in tids])

    Y = get_Y(objclass,z,z_conf,qso_zlim=z_lim,mode=mode)

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    print("INFO: {} spectra in returned dataset".format(tids.shape[0]))

    if return_spid:
        return tids,X,Y,z,bal,spid0,spid1,spid2

    return tids,X,Y,z,bal

def get_Y(objclass,z,z_conf,qso_zlim=2.1,mode='BOSS'):

    Y = np.zeros((objclass.shape[0],5))

    if mode == 'BOSS':
        ## STAR
        w = (objclass==1) & (z_conf==3)
        Y[w,0] = 1

        ## GALAXY
        w = (objclass==4) & (z_conf==3)
        Y[w,1] = 1

        ## QSO_LZ
        w = ((objclass==3) | (objclass==30)) & (z<qso_zlim) & (z_conf==3)
        Y[w,2] = 1

        ## QSO_HZ
        w = ((objclass==3) | (objclass==30)) & (z>=qso_zlim) & (z_conf==3)
        Y[w,3] = 1

        ## BAD
        w = (z_conf != 3)
        Y[w,4] = 1
    elif mode == 'DESI':
        ## STAR
        w = ((objclass=='STAR') | (objclass=='WD')) & (z_conf==4)
        Y[w,0] = 1

        ## GALAXY
        w = (objclass=='GALAXY') & (z_conf==4)
        Y[w,1] = 1

        ## QSO_LZ
        w = (objclass=='QSO') & (z<qso_zlim) & (z_conf==4)
        Y[w,2] = 1

        ## QSO_HZ
        w = (objclass=='QSO') & (z>=qso_zlim) & (z_conf==4)
        Y[w,3] = 1

        ## BAD
        w = (z_conf != 4)
        Y[w,4] = 1
    elif mode == 'DESISIM':
        ## STAR
        w = ((objclass=='STAR') | (objclass=='WD')) & (z_conf==4)
        Y[w,0] = 1

        ## GALAXY
        w = (objclass=='GALAXY') & (z_conf==4)
        Y[w,1] = 1

        ## QSO_LZ
        w = (objclass=='QSO') & (z<qso_zlim) & (z_conf==4)
        Y[w,2] = 1

        ## QSO_HZ
        w = (objclass=='QSO') & (z>=qso_zlim) & (z_conf==4)
        Y[w,3] = 1

        ## BAD
        w = (z_conf != 4)
        Y[w,4] = 1

    return Y

################################################################################
## Training functions.

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
## Export data.

# TODO: what does this do exactly? Search for uses.
# Doesn't seem to be used in the repo at all.
# Maybe in scripts? Haven't seen any exts with 'DATA' as a name...
def export_data(fout,tids,data):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(data,extname="DATA")
    tids = np.array(tids)
    h.write([tids],names=["TARGETID"],extname="METADATA")
    h.close()
