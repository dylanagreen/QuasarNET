from __future__ import print_function

from astropy.io import fits
import numpy as np
from numpy import random
import fitsio
from random import randint
from os.path import dirname, isfile
import glob

from quasarnet import utils

# TODO: move this somewhere else? Feels an odd place to keep it?
# Made a class utils.Wave to replace this. Has these llmin/llmax/dll as
# default but can add functionality to overwrite this with relative ease. Left
# without for now to avoid any potential issues.

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

    objclass_codes = codify_objclass(data['OBJCLASS'],mode)
    data['OBJCLASS'] = objclass_codes

    zconf_codes = codify_zconf(data['Z_CONF'],mode)
    data['Z_CONF'] = zconf_codes

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
def read_spcframe(b_spcframe, r_spcframe, fibers, verbose=False,
                llmin=np.log10(3600.), llmax=np.log10(10000.), dll=1.e-3,
                nmasked_max=None):

    '''
    reads data from spcframes

    Arguments:
        b_spcframe -- spcframe from b part of spectrograph
        r_spcframe -- spcframe from r part of spectrograph

    Returns:
        fids -- fiberids
        fliv -- flux/iv
    '''

    hb = fitsio.FITS(b_spcframe)
    hr = fitsio.FITS(r_spcframe)

    ## Read the data by horizontally stacking.
    fl_aux = np.hstack([hb[0].read(), hr[0].read()])
    iv_aux = np.hstack([hb[1].read()*((hb[2].read()&2**25)==0),
        hr[1].read()*((hr[2].read()&2**25)==0)])
    wave_aux = np.hstack([10**hb[3].read(), 10**hr[3].read()])

    ### HACK
    # Full mask option
    #iv_aux = np.hstack([hb[1].read()*(hb[2].read()==0),
    #    hr[1].read()*(hr[2].read()==0)])
    # No mask option
    #iv_aux = np.hstack([hb[1].read(),hr[1].read()])
    ###

    ## Filter the data by those we're interested in.
    plate = hb[0].read_header()["PLATEID"]
    fids = hb[5]["FIBERID"][:]
    wqso = np.in1d(fids, fibers)
    fids = fids[wqso]
    fl_aux = fl_aux[wqso,:]
    iv_aux = iv_aux[wqso,:]
    wave_aux = wave_aux[wqso,:]
    if verbose:
        print("INFO: found {} quasars in file {}".format(wqso.sum(),b_spcframe))

    ## Construct the output grids for flux and iv.
    nspec = len(fids)
    wave_out = utils.Wave(llmin=llmin,llmax=llmax,dll=dll)
    fl = np.zeros((nspec,wave_out.nbins))
    iv = np.zeros((nspec,wave_out.nbins))

    for i in range(nspec):

        ## Calculate how to rebin the data.
        bins, w = utils.rebin_wave(wave_aux[i,:],wave_out)
        bins = bins[w]

        fl_spec = fl_aux[:,w][i]
        iv_spec = iv_aux[:,w][i]

        ## Rebin the flux and iv and add them to the pre-constructed grids.
        c = np.bincount(bins,weights=fl_spec*iv_spec)
        fl[i,:len(c)] += c
        c = np.bincount(bins,weights=iv_spec)
        iv[i,:len(c)] += c

    ## Normalise the flux and stack fl and iv.
    w = iv>0
    fl[w] /= iv[w]
    fliv = np.hstack((fl,iv))

    ## Filter out spectra with too many bad pixels.
    wbad = (iv==0)
    if nmasked_max is None:
        nmasked_max = len(wave_out.wave_grid)+1
    w = (wbad.sum(axis=1)>nmasked_max)
    if verbose:
        print('INFO: rejecting {} spectra with too many bad pixels'.format(w.sum()))
    if (~w).sum()==0:
        return None
    fids = fids[~w]
    fliv = fliv[~w,:]

    assert ~np.isnan(fliv).any()

    return fids, fliv

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
    #spall = fitsio.FITS(spall)
    #plate = spall[1]["PLATE"][:]
    #mjd = spall[1]["MJD"][:]
    #fid = spall[1]["FIBERID"][:]
    #tid = spall[1]["THING_ID"][:].astype(int)
    #specprim = spall[1]["SPECPRIMARY"][:]

    ## Much faster in astropy.
    spall = fits.open(spall)
    plate = spall[1].data['PLATE']
    mjd = spall[1].data['MJD']
    fid = spall[1].data['FIBERID']
    tid = spall[1].data['THING_ID'].astype(int)
    specprim = spall[1].data['SPECPRIMARY']

    ## Construct a dictionary mapping plate, mjd and fiberid to thing_id.
    pmf2tid = {(p,m,f):t for p,m,f,t,s in zip(plate,mjd,fid,tid,specprim)}
    spall.close()

    return tid, pmf2tid

## Read the spectra files from BOSS (spplate) or DESI (desi_spectra)
def read_spplate(fin, fibers, verbose=False, llmin=np.log10(3600.), llmax=np.log10(10000.), dll=1.e-3, nmasked_max=None):

    '''
    Given an spPlate file, returns spectra, interpolated onto a new wavelength
    grid.

    input:
        -- fin: path to spplate file to read
        -- fibers: list of fiberids to read
        -- verbose: whether to print debugging statements or not
        -- llmin: min value of log10(wavelength) to use in wavelength grid
        -- llmax: max value of log10(wavelength) to use in wavelength grid
        -- dll: value of distance between log10(wavelength) pixels to use in
            wavelength grid
    output: thid, data
        -- fids: list of fiberids in output
        -- data: flux/iv
    '''

    ## Open the file and read data from the header.

    ## Want to switch to astropy.io.fits here, as fitsio crashes when certain files are used.
    ## For example, the header keyname 'EXPID**' in /global/projecta/projectdirs/sdss/data/sdss/dr13/eboss/spectro/redux/v5_9_0/6138/spPlate-6138-56598.fits
    h = fitsio.FITS(fin)
    try:
        head = h[0].read_header()
    except OSError:
        print('WARNING: problem with reading headers in {} with fitsio'.format(fin))
        print('WARNING: Using astropy instead')
        temp = fits.open(fin)
        head = temp[0].header
        temp.close()

    c0 = head["COEFF0"]
    c1 = head["COEFF1"]
    p = head["PLATEID"]
    m = head["MJD"]

    ## Filter the fiberids in the file by those we're interested in.
    fids = h[5]["FIBERID"][:]
    wqso = np.in1d(fids, fibers)
    fids = fids[wqso]
    if verbose:
        print("INFO: found {} quasars in file {}".format(wqso.sum(),b_spcframe))

    ## Read the data from file.
    fl_aux = h[0].read()[wqso,:]
    iv_aux = h[1].read()[wqso,:]*((h[2].read()[wqso]&2**25)==0)

    ## Construct the grids for flux and iv.
    nspec = len(fibers)
    wave_out = utils.Wave(llmin=llmin,llmax=llmax,dll=dll)
    fl = np.zeros((nspec, wave_out.nbins))
    iv = np.zeros((nspec, wave_out.nbins))

    ## Calculate how to rebin the data.
    wave_grid = 10**(c0 + c1*np.arange(fl_aux.shape[1]))
    bins, w = utils.rebin_wave(wave_grid,wave_out)
    bins = bins[w]
    fl_aux = fl_aux[:,w]
    iv_aux = iv_aux[:,w]

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
    fliv = np.hstack((fl,iv))

    ## Filter out spectra with too many bad pixels.
    wbad = (iv==0)
    if nmasked_max is None:
        nmasked_max = len(wave_out.wave_grid)+1
    w = (wbad.sum(axis=1)>nmasked_max)
    if verbose:
        print('INFO: rejecting {} spectra with too many bad pixels'.format(w.sum()))
    if (~w).sum()==0:
        return None
    fids = fids[~w]
    fliv = fliv[~w,:]

    assert ~np.isnan(fliv).any()

    return fids, fliv

## ??
def read_single_exposure(fin, fibers, verbose=False, best_exp=True, random_exp=False, random_seed=0, llmin=np.log10(3600.), llmax=np.log10(10000.), dll=1.e-3):
    '''
    Given an spPlate file, returns spectra from one exposure that went into
    that file, interpolated onto a new wavelength grid.

    input:
        -- fin: path to spplate file to read
        -- fibers: list of fiberids to read
        -- verbose: whether to print debugging statements or not
        -- best_exp: read only the "BESTEXP" exposure from all the available
            exposures (determined by minimum S/R^2?)
        -- random_exp: read only one random exposure from all the available
            exposures
        -- random_seed: seed for choosing random exposure
        -- llmin: min value of log10(wavelength) to use in wavelength grid
        -- llmax: max value of log10(wavelength) to use in wavelength grid
        -- dll: value of distance between log10(wavelength) pixels to use in
            wavelength grid
    output: thid, data
        -- fids: list of fiberids in output
        -- fliv: flux/iv
    '''

    ## Want to switch to astropy.io.fits here, as fitsio crashes when certain files are used.
    ## For example, the header keyname 'EXPID**' in /global/projecta/projectdirs/sdss/data/sdss/dr13/eboss/spectro/redux/v5_9_0/6138/spPlate-6138-56598.fits
    spplate = fitsio.FITS(fin)
    try:
        head = spplate[0].read_header()
    except OSError:
        print('WARNING: problem with reading headers in {} with fitsio'.format(fin))
        print('WARNING: Using astropy instead')
        temp = fits.open(fin)
        head = temp[0].header
        temp.close()

    spcframes = []
    spectros = ['1','2']

    if best_exp:
        path = dirname(fin)

        bestexp = head["BESTEXP"]
        expid = str(bestexp).zfill(8)
        for s in spectros:
            b_exp = path+"/spCFrame-b"+s+'-'+expid+".fits"
            r_exp = path+"/spCFrame-r"+s+'-'+expid+".fits"
            if (isfile(b_exp) and isfile(r_exp)):
                spcframes.append((b_exp,r_exp))
            else:
                print('WARN: at least one of {} and {} do not exist.'.format(b_exp,r_exp))

        if verbose:
            print("INFO: using best exposure",expid)
    elif random_exp:
        path = dirname(fin)

        ## For each exposure that went into the spplate file, extract the
        ## expid from the spplate header. Remove duplicates (from different
        ## cameras) and put into a random order. Use [plate,mjd,random_seed]
        ## as a random seed.
        nexp = head["NEXP"]
        # Set nexp to at most 99 as larger numbers do not fit format.
        nexp = min(nexp,99)
        expids = list(set([head["EXPID"+str(n+1).zfill(2)][3:11] for n in range(nexp)]))
        expids.sort()
        gen = np.random.RandomState(seed=[head["PLATEID"],head["MJD"],random_seed])
        gen.shuffle(expids)

        ### HACK
        #expids = ['00104927']
        ###

        ## For each expid:
        ind = 0
        exit = False
        nspcframe_pairs = 2
        while (ind<len(expids)) and (not exit):
            expid = expids[ind]
            ind += 1

            # Check that this exposure exists for all cameras.
            spcframe_pairs_found = 0
            for s in spectros:
                b_exp = path+"/spCFrame-b"+s+'-'+expid+".fits"
                r_exp = path+"/spCFrame-r"+s+'-'+expid+".fits"
                if (isfile(b_exp) and isfile(r_exp)):
                    spcframe_pairs_found += 1

            if spcframe_pairs_found<nspcframe_pairs:
                print('WARN: only {} spcframe pairs found for expid {} in {}, moving to next exp...'.format(spcframe_pairs_found,fin,expid))
            
            # If so, add exposures to the list of infiles.
            if spcframe_pairs_found==nspcframe_pairs:
                for s in spectros:
                    b_exp = path+"/spCFrame-b"+s+'-'+expid+".fits"
                    r_exp = path+"/spCFrame-r"+s+'-'+expid+".fits"
                    if (isfile(b_exp) and isfile(r_exp)):
                        spcframes.append((b_exp,r_exp))

                # Exit the while loop.
                exit = True

            # If not, only require spcframe files from one spectrograph.
            elif (ind==len(expids)) and (nspcframe_pairs==2):
                ind = 0
                nspcframe_pairs = 1

        # If we did not find files, print a notification.
        if spcframe_pairs_found==0:
            print("WARN: could not find any b/r pairs of spCFrame files for any single exposure in spplate {}".format(fin))
            return None
        elif spcframe_pairs_found==1:
            print("WARN: could only find 1 b/r pair of spCFrame files for any single exposure in spplate {}".format(fin))
        else:
            if verbose:
                print("INFO: using randomly chosen exposure",expid)

    fids = []
    fliv = []
    #print(spcframes)
    for spcframe in spcframes:
        aux = read_spcframe(spcframe[0], spcframe[1], fibers, verbose=False,
            llmin=llmin, llmax=llmax, dll=dll)
        if aux is not None:
            fids.append(aux[0])
            fliv.append(aux[1])

    fids = np.concatenate(fids)
    fliv = np.vstack(fliv)

    return fids, fliv

def read_desi_spectra_list(fin, ignore_quasar_mask=False, verbose=True, period='survey', cmx_bitname='MINI_SV_QSO', llmin=np.log10(3600.), llmax=np.log10(9800.), dll=1.e-3, mode='DESI'):

    '''
    reads data from DESI spectra files (per HEALPix pixel)

    Arguments:
        fin -- list of spectra files to read
        ignore_quasar_mask -- include all spectra, not just quasar targets
        verbose -- chatty or not
        period -- which period of DESI are we in (decides which targeting bits to use)

    Returns:
        meta -- dictionary of metadata
        fl -- hstacked flux/iv array
    '''

    # Obtain the mask from desitarget if available.
    if ignore_quasar_mask:
        tb = None
    else:
        tb = utils.get_targeting_bits('DESI',verbose=False,desi_period=period,desi_cmx_bitname=cmx_bitname)

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

        aux = read_desi_spectra(f, tb, verbose=verbose, llmin=llmin, llmax=llmax, dll=dll, mode=mode)

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
                    tids = tids[~check]
                    spid0 = spid0[~check]
                    spid1 = spid1[~check]
                    spid2 = spid2[~check]
                    fl = fl[~check,:]
                    iv = iv[~check,:]

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
        print("INFO: found {} good spectra".format(nspec))
        return tids, spid0, spid1, spid2, fliv

    else:
        print('WARN: no quasar spectra found in given file list.')
        return None

def read_desi_spectra(f, tb, verbose=True, llmin=np.log10(3600.), llmax=np.log10(9800.), dll=1.e-3, mode='DESI'):

    tid_field = utils.get_tid_field(mode)
    spid_fields = utils.get_spectrum_id_fields(mode)

    h = fitsio.FITS(f)

    wqso = np.zeros(len(h[1][:])).astype('bool')
    if tb is not None:
        for kw,val in tb.items():
            mask = sum([2**b for b in val])
            wqso_kw = ((mask & h[1][kw][:])>0)
            print('INFO: found {} quasar targets with target bits {} in {}'.format(wqso_kw.sum(),val,kw))
            wqso |= wqso_kw

    nspec_init = wqso.sum()
    if nspec_init == 0: return None

    if verbose:
        print("INFO: found {} target spectra".format(nspec_init))

    tids = h[1][tid_field['TARGETID']][:][wqso]
    spid0 = h[1][spid_fields['SPID0']][:][wqso]
    spid1 = h[1][spid_fields['SPID1']][:][wqso]
    spid2 = h[1][spid_fields['SPID2']][:][wqso]

    ## Here, we determine what to do with spectra that duplicate in some way.
    met = [(t,s0,s1,s2) for t,s0,s1,s2 in zip(tids,spid0,spid1,spid2)]
    #umet = [tuple(x) for x in np.unique(met,axis=0)]
    umet = set(met)

    ## Remove any entries with duplicated metadata.
    wdup = np.zeros(tids.shape).astype('bool')
    for um in umet:
        j = np.where([um==M for M in met])[0]
        if len(j)>1:
            wdup[j[0]] = 1
            if verbose:
                print('WARN: (tid,spid)={} is duplicated'.format(um))
        else:
            wdup[j] = 1

    tids = tids[wdup]
    spid0 = spid0[wdup]
    spid1 = spid1[wdup]
    spid2 = spid2[wdup]

    wave_out = utils.Wave(llmin=llmin,llmax=llmax,dll=dll)

    nspec = len(tids)
    fl = np.zeros((nspec, wave_out.nbins))
    iv = np.zeros((nspec, wave_out.nbins))

    if nspec == 0: return None

    bands = [x.get_extname().split('_')[0] for x in h.hdu_list if 'WAVELENGTH' in x.get_extname()]
    for band in bands:
        h_wave = h["{}_WAVELENGTH".format(band)].read()

        bins, w = utils.rebin_wave(h_wave,wave_out)
        h_wave = h_wave[w]
        bins = bins[w]

        # Filter by valid pixels and QSOs.
        fl_aux = h["{}_FLUX".format(band)].read()[:,w]
        iv_aux = h["{}_IVAR".format(band)].read()[:,w]
        fl_aux = fl_aux[wqso][wdup]
        iv_aux = iv_aux[wqso][wdup]

        # Set NaN values of flux to zero, and make sure they have ivar zero.
        w_nan = np.isnan(fl_aux)
        fl_aux[w_nan] = 0.
        iv_aux[w_nan] = 0.

        ivfl_aux = fl_aux*iv_aux

        for i,t in enumerate(tids):
            c = np.bincount(bins, weights = ivfl_aux[i,:])
            fl[i,:len(c)] += c
            c = np.bincount(bins, weights = iv_aux[i,:])
            iv[i,:len(c)] += c

    # Normalise flux by dividing by summed ivars.
    w = iv>0
    fl[w] /= iv[w]

    return tids, spid0, spid1, spid2, fl, iv

# TODO: write this.
def read_bal_data_drq(drq, mode='BOSS'):

    return bal_flag, bi_civ

## Simulated DESI specific functions.
def read_bal_data_desisim(truth_files,bal_templates):
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
    tids = []
    bal_templateid = []
    for f in truth_files:
        h = fitsio.FITS(f)

        """
        ## Needed for qq runs atm
        tids.append(h['QSO_META']['TARGETID'][:])
        bal_templateid.append(h['QSO_META']['BAL_TEMPLATEID'][:])
        """
        tids.append(h['TRUTH_QSO']['TARGETID'][:])
        bal_templateid.append(h['TRUTH_QSO']['BAL_TEMPLATEID'][:])
        h.close()

    tids = np.hstack(tids)
    bal_templateid = np.hstack(bal_templateid)

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

def read_truth_desisim(truth_files):

    tid_field = utils.get_tid_field('DESISIM')
    truth_fields = utils.get_truth_fields('DESISIM')

    tr_dict = {}
    tr_dict[tid_field['TARGETID']] = []
    for k in truth_fields.keys():
        tr_dict[k] = []

    for f in truth_files:

        h = fitsio.FITS(f)

        tr_dict[tid_field['TARGETID']].append(h[1][tid_field['TARGETID']][:])
        for k in truth_fields.keys():
            tr_dict[k].append(h[1][truth_fields[k]][:])

        h.close()

    tr_dict[tid_field['TARGETID']] = np.hstack(tr_dict[tid_field['TARGETID']])
    for k in truth_fields.keys():
        tr_dict[k] = np.hstack(tr_dict[k])

    ## For a simulation, we have an absolute truth, and so add "confidence"
    ## artificially.
    tr_dict['Z_CONF'] = 4*np.ones_like(tr_dict['Z'])

    ## OBJCLASS is a string, and has trailing spaces (fitsio related).
    ## We remove them here to avoid confusion.
    tr_dict['OBJCLASS'] = np.array([y.strip(' ') for y in tr_dict['OBJCLASS']])

    objclass_codes = codify_objclass(tr_dict['OBJCLASS'],'DESISIM')
    tr_dict['OBJCLASS'] = objclass_codes

    zconf_codes = codify_zconf(tr_dict['Z_CONF'],'DESISIM')
    tr_dict['Z_CONF'] = zconf_codes

    return tr_dict

def read_targets_desisim(targets,period):

    tid_field = utils.get_tid_field('DESISIM')
    targeting_bit_col = utils.get_desi_targeting_bit_col(period)

    h = fitsio.FITS(targets)

    ta_dict = {}
    ta_dict[tid_field['TARGETID']] = h[1][tid_field['TARGETID']][:]
    ta_dict[targeting_bit_col] = h[1][targeting_bit_col][:]

    h.close()

    return ta_dict

def codify_objclass(objclass,mode):

    ## Get a dictionary mapping the coded classes to a list of the native class
    ## names included in that class.
    cc_dict = utils.get_class_codes(mode)

    ## For each class code, find which objects have native class names that fall
    ## into its class.
    objclass_codes = np.zeros(objclass.shape)
    for k in cc_dict.keys():
        w = [objclass==classname for classname in cc_dict[k]]
        w = np.any(w,axis=0)
        objclass_codes[w] = k

    return objclass_codes.astype('i4')

def codify_zconf(zconf,mode):

    ## Get a dictionary mapping the coded classes to a list of the native class
    ## names included in that class.
    zc_dict = utils.get_zconf_codes(mode)

    ## For each class code, find which objects have native class names that fall
    ## into its class.
    zconf_codes = np.zeros(zconf.shape)
    for k in zc_dict.keys():
        w = [zconf==classname for classname in zc_dict[k]]
        w = np.any(w,axis=0)
        zconf_codes[w] = k

    return zconf_codes.astype('i4')

"""## Not used as far as I can see.
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
"""
################################################################################
## Read data after parsing.

def read_truth(fi,mode='BOSS'):
    '''
    reads a list of truth files and returns a truth dictionary

    Arguments:
        fi -- list of truth files (list of string)

    Returns:
        truth -- dictionary of THING_ID: truth data instance

    '''

    class metadata:
        pass

    cols = list(utils.get_truth_fields(None).keys()) + list(utils.get_bal_fields(None).keys())

    truth_cols = list(utils.get_truth_fields(None).keys())
    bal_cols = list(utils.get_bal_fields(None).keys())

    BOSS_tf_dict = utils.get_truth_fields(mode)
    BOSS_bf_dict = utils.get_bal_fields(mode)

    truth = {}

    ## Cycle through the files, extracting data from each one.
    for f in fi:

        # Open the file and get the tids.
        h = fitsio.FITS(f)
        try:
            tids = h[1]['TARGETID'][:]
        except ValueError:
            tids = h[1]['THING_ID'][:]

        try:
            truth_cols_dict = {c.lower():h[1][c][:] for c in truth_cols}
        except ValueError:
            truth_cols_dict = {c.lower():h[1][BOSS_tf_dict[c]][:] for c in truth_cols}
            truth_cols_dict['objclass'] = codify_objclass(truth_cols_dict['objclass'],mode)
            truth_cols_dict['z_conf'] = codify_zconf(truth_cols_dict['z_conf'],mode)

        try:
            bal_cols_dict = {c.lower():h[1][c][:] for c in bal_cols}
        except ValueError:
            bal_cols_dict = {c.lower():h[1][BOSS_bf_dict[c]][:] for c in bal_cols}

        # Cycle through each tid.
        for i,t in enumerate(tids):
            m = metadata()
            for c in truth_cols_dict.keys():
                setattr(m,c,truth_cols_dict[c][i])
            for c in bal_cols_dict.keys():
                setattr(m,c,bal_cols_dict[c][i])
            truth[t] = m

        h.close()

    return truth

def read_data(fi, truth=None, z_lim=2.1, return_spid=False, nspec=None, verbose=True):
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
        if verbose:
            print('INFO: reading data from {}'.format(f))
        h = fitsio.FITS(f)
        if nspec is None:
            nspec = h[1].get_nrows()
        aux_tids = h[1]['TARGETID'][:nspec].astype(int)
        if verbose:
            print("INFO: found {} spectra in file {}".format(aux_tids.shape[0], f))

        ## remove thing_id == -1 or not in sdrq
        w = (aux_tids != -1)
        if verbose:
            print("INFO: removing {} spectra with thing_id=-1".format((~w).sum()),flush=True)
        aux_tids = aux_tids[w]
        aux_X = h[0][:nspec,:]
        aux_X = aux_X[w,:]

        if truth is not None:
            w_in_truth = np.in1d(aux_tids, list(truth.keys()))
            if verbose:
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

    ## Get the number of cells.
    ncells = X.shape[1]/2.
    assert ncells==round(ncells)
    ncells = round(ncells)
    if verbose:
        print('INFO: Spectra have {} cells'.format(ncells))

    we = X[:,ncells:]
    w = we.sum(axis=1)==0
    if verbose:
        print("INFO: removing {} spectra with zero weights".format(w.sum()))
    X = X[~w]
    tids = tids[~w]

    if return_spid:
        spid0 = spid0[~w]
        spid1 = spid1[~w]
        spid2 = spid2[~w]

    mdata = np.average(X[:,:ncells], weights = X[:,ncells:], axis=1)
    sdata = np.average((X[:,:ncells]-mdata[:,None])**2,
            weights = X[:,ncells:], axis=1)
    sdata=np.sqrt(sdata)

    w = sdata == 0
    if verbose:
        print("INFO: removing {} spectra with zero flux".format(w.sum()))
    X = X[~w]
    tids = tids[~w]
    mdata = mdata[~w]
    sdata = sdata[~w]

    if return_spid:
        spid0 = spid0[~w]
        spid1 = spid1[~w]
        spid2 = spid2[~w]

    X = X[:,:ncells]-mdata[:,None]
    X /= sdata[:,None]

    if truth==None:
        if return_spid:
            return tids,X,spid0,spid1,spid2
        else:
            return tids,X

    ## remove zconf == 0 (not inspected)
    observed = [(truth[t].objclass>0) or (truth[t].z_conf>0) for t in tids]
    observed = np.array(observed, dtype=bool)
    if verbose:
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

    Y = get_Y(objclass,z,z_conf,qso_zlim=z_lim)

    ## check that all spectra have exactly one classification
    assert (Y.sum(axis=1).min()==1) and (Y.sum(axis=1).max()==1)

    if verbose:
        print("INFO: {} spectra in returned dataset".format(tids.shape[0]))

    if return_spid:
        return tids,X,Y,z,bal,spid0,spid1,spid2

    return tids,X,Y,z,bal

def get_Y(objclass,z,z_conf,qso_zlim=2.1):

    Y = np.zeros((objclass.shape[0],5))

    ## STAR
    w = (objclass==1) & (z_conf==2)
    Y[w,0] = 1

    ## GALAXY
    w = (objclass==2) & (z_conf==2)
    Y[w,1] = 1

    ## QSO_LZ
    w = (objclass==3) & (z<qso_zlim) & (z_conf==2)
    Y[w,2] = 1

    ## QSO_HZ
    w = (objclass==3) & (z>=qso_zlim) & (z_conf==2)
    Y[w,3] = 1

    ## BAD
    w = (z_conf != 2)
    Y[w,4] = 1

    return Y

################################################################################
## Training functions.

# TODO: should this go in utils maybe?
from .utils import absorber_IGM
from scipy.interpolate import interp1d
def box_offset(z, line='LYA', nboxes = 13, llmin=np.log10(3600.), llmax=np.log10(10000.), dll=1.e-3):

    wave = utils.Wave(llmin=llmin,llmax=llmax,dll=dll)

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
def objective(z, Y, bal, lines=['LYA'], lines_bal=['CIV(1548)'], nboxes=13,
              llmin=np.log10(3600.), llmax=np.log10(10000.), dll=1.e-3):

    box=[]
    sample_weight = []
    for l in lines:
        # TODO: Do we need to use "weight_line"?
        box_line, offset_line, weight_line = box_offset(z,
                line = l, nboxes=nboxes, llmin=llmin, llmax=llmax, dll=dll)

        w = (Y.argmax(axis=1)==2) | (Y.argmax(axis=1)==3)
        ## set to zero where object is not a QSO
        ## (the line confidence should be zero)
        box_line[~w]=0
        box.append(np.concatenate([box_line, offset_line], axis=-1))
        sample_weight.append(np.ones(Y.shape[0]))

    for l in lines_bal:
        box_line, offset_line, weight_line = box_offset(z,
                line = l, nboxes=nboxes, llmin=llmin, llmax=llmax, dll=dll)

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
