from scipy.interpolate import interp1d
from numpy import zeros, arange, array
import numpy as np

def process_preds(preds, lines, lines_bal, wave=None):
    '''
    Convert network predictions to c_lines, z_lines and z_best

    Arguments:
    preds: float, array
        model predictions, output of model.predict

    lines: string, array
        list of line names

    lines_bal: string, array
        list of BAL line names

    Returns:
    c_line: float, array
        line confidences, shape: (nlines, nspec)
    z_line: float, array
        line redshifts, shape: (nlines, nspec)
    zbest: float, array
        redshift of highest confidence line, shape: (nspec)
    c_line_bal: float, array
        line confidences of BAL lines, shape: (nlines_bal, nspec)
    z_line_bal: float, array
        line redshfits of BAL lines, shape: (nlines_bal, nspec)
    '''

    # Ensure that the number of lines + bal lines is consistent with the size
    # of the predictions array.
    assert len(lines)+len(lines_bal)==len(preds)
    nspec, nboxes = preds[0].shape
    nboxes //= 2
    nlines = len(lines)
    print('INFO: nspec = {}, nboxes={}'.format(nspec, nboxes))

    # Set up the output matrices.
    c_line = zeros((nlines, nspec))
    z_line = zeros((nlines, nspec))

    # For each line, fill in the output array. Done for all spectra at once.
    for il in range(len(lines)):
        c_line[il], z_line[il] = line_preds_to_properties(preds[il],lines[il],wave=wave)

        """
        ## Rest frame wavelength of line.
        l = absorber_IGM[lines[il]]
        ## Index of most likely box.
        j = preds[il][:,:13].argmax(axis=1)
        ## Offset within this box.
        offset = preds[il][arange(nspec, dtype=int), nboxes+j]
        ## Put the confidence and redshift for the line into the output array.
        c_line[il] = preds[il][:,:13].max(axis=1)
        z_line[il] = i_to_wave((j+offset)*len(wave)/nboxes)/l - 1
        """

    # Get the best estimate of z from the most confident line
    zbest = z_line[c_line.argmax(axis=0),arange(nspec)]
    zbest = array(zbest)

    nlines_bal = len(lines_bal)
    c_line_bal=zeros((nlines_bal, nspec))
    z_line_bal=zeros((nlines_bal, nspec))

    for il in range(len(lines_bal)):
        c_line_bal[il], z_line_bal[il] = line_preds_to_properties(preds[nlines+il],lines_bal[il],wave=wave)

        """
        l = absorber_IGM[lines_bal[il]]
        j = preds[nlines+il][:,:nboxes].argmax(axis=1)
        offset = preds[nlines+il][arange(nspec, dtype=int), nboxes+j]
        c_line_bal[il] = preds[il+nlines][:,:nboxes].max(axis=1)
        z_line_bal[il] = i_to_wave((j+offset)*len(wave)/nboxes)/l-1
        """

    return c_line, z_line, zbest, c_line_bal, z_line_bal

def line_preds_to_properties(line_preds,line,wave=None):
    '''
    Convert network predictions for 1 line to c_line, z_line.

    Arguments:
    line_preds: float, array
        model predictions for one line, output of model.predict

    line: string
        line name

    Returns:
    c_line: float, array
        line confidences, shape: (nspec, )
    z_line: float, array
        line redshifts, shape: (nspec, )
    '''

    nspec, nboxes = line_preds.shape
    nboxes //= 2

    # Construct an interpolator to go from the index along a wave vector to the
    # wavelength associated with this position.
    if wave is None:
        print('WARN: No wave grid information provided; using default:')
        wave = Wave()
        print('      - lmin={}, lmax={}, dll={}'.format(10**wave.llmin,10**wave.llmax,wave.dll))
    i_to_wave = interp1d(arange(len(wave.wave_grid)), wave.wave_grid,
            bounds_error=False, fill_value='extrapolate')

    # Fill in the output array. Done for all spectra at once.
    ## Rest frame wavelength of line.
    l = absorber_IGM[line]
    ## Index of most likely box.
    j = line_preds[:,:nboxes].argmax(axis=1)
    ## Offset within this box.
    offset = line_preds[arange(nspec, dtype=int), nboxes+j]
    ## Put the confidence and redshift for the line into the output array.
    c_line = line_preds[:,:nboxes].max(axis=1)
    z_line = i_to_wave((j+offset)*len(wave.wave_grid)/nboxes)/l - 1

    return c_line, z_line

class Wave:
    def __init__(self,llmin=np.log10(3600.),llmax=np.log10(10000.),dll=1.e-3):

        self.llmin = llmin
        self.llmax = llmax
        self.dll = dll

        nbins = int((llmax-llmin)/dll)
        wave_grid = 10**(llmin + np.arange(nbins)*dll)
        self.nbins = nbins
        self.wave_grid = wave_grid

        return

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

def get_tid_field(mode):

    tid_field = {}

    if mode == 'BOSS':
        tid_field['TARGETID'] = 'THING_ID'

    elif mode == 'DESI':
        tid_field['TARGETID'] = 'TARGETID'

    elif mode == 'DESISIM':
        tid_field['TARGETID'] = 'TARGETID'

    elif mode == None:
        tid_field['TARGETID'] = ''

    return tid_field

def get_spectrum_id_fields(mode):

    spid_fields = {}

    if mode == 'BOSS':
        spid_fields['SPID0'] = 'PLATE'
        spid_fields['SPID1'] = 'MJD'
        spid_fields['SPID2'] = 'FIBERID'

    elif mode == 'DESI':
        spid_fields['SPID0'] = 'TILEID'
        spid_fields['SPID1'] = 'NIGHT'
        spid_fields['SPID2'] = 'FIBER'

    elif mode == 'DESISIM':
        spid_fields['SPID0'] = 'FIBER'#'TILEID'
        spid_fields['SPID1'] = 'FIBER'#'NIGHT'
        spid_fields['SPID2'] = 'FIBER'

    elif mode == None:
        spid_fields['SPID0'] = ''
        spid_fields['SPID1'] = ''
        spid_fields['SPID2'] = ''

    return spid_fields

def get_truth_fields(mode):

    truth_fields = {}

    if mode == 'BOSS':
        truth_fields['Z'] =         'Z_VI'
        truth_fields['OBJCLASS'] =  'CLASS_PERSON'
        truth_fields['Z_CONF'] =    'Z_CONF_PERSON'

    elif mode == 'DESI':
        truth_fields['Z'] =         'Z_VI'
        truth_fields['OBJCLASS'] =  'CLASS_PERSON'
        truth_fields['Z_CONF'] =    'Z_CONF_PERSON'

    elif mode == 'DESISIM':

        truth_fields['Z'] =         'TRUEZ'
        truth_fields['OBJCLASS'] =  'TRUESPECTYPE'
        """
        ## Needed for qq atm
        truth_fields['Z'] =         'Z'
        truth_fields['OBJCLASS'] =  'OBJTYPE'
        """

    elif mode == None:
        truth_fields['Z'] =         ''
        truth_fields['OBJCLASS'] =  ''
        truth_fields['Z_CONF'] =    ''

    return truth_fields

def get_bal_fields(mode):

    bal_fields = {}

    if mode == 'BOSS':
        bal_fields['BAL_FLAG'] = 'BAL_FLAG_VI'
        bal_fields['BI_CIV'] =   'BI_CIV'

    elif mode == 'DESI':
        bal_fields['BAL_FLAG'] = 'BAL_FLAG_VI'
        bal_fields['BI_CIV'] =   'BI_CIV'

    elif mode == 'DESISIM':
        bal_fields['BAL_FLAG'] = 'BAL_FLAG_VI'
        bal_fields['BI_CIV'] =   'BI_CIV'

    elif mode == None:
        bal_fields['BAL_FLAG'] =  ''
        bal_fields['BI_CIV'] =    ''

    return bal_fields

def get_quasar_mask(verbose=True,period='survey'):

    try:
        import desitarget
        if period=='survey':
            quasar_mask = desitarget.targetmask.desi_mask.mask('QSO')
        elif period=='sv':
            quasar_mask = desitarget.sv1.sv1_targetmask.desi_mask.mask('QSO')
        elif period=='cmx':
            quasar_mask = desitarget.cmx.cmx_targetmask.cmx_mask.mask('MINI_SV_QSO')
    except ImportError:
        if verbose:
            print("WARN: can't load desi_mask, using hardcoded targetting value!")
        if period=='survey':
            quasar_mask = 2**2
        elif period=='sv':
            quasar_mask = 2**2
        elif period=='cmx':
            quasar_mask = 2**55

    return quasar_mask

def get_class_codes(mode):

    ## Unobserved is 0
    ## Star or WD is 1
    ## Galaxy is 2
    ## QSO is 3

    class_codes = {}

    if mode == 'BOSS':

        class_codes[0] = [0]
        class_codes[1] = [1]
        class_codes[2] = [4]
        class_codes[3] = [3,30]

    elif mode == 'DESI':

        # TODO: Not sure about 0 here.
        class_codes[0] = ['']
        class_codes[1] = ['STAR','WD']
        class_codes[2] = ['GALAXY']
        class_codes[3] = ['QSO']

    elif mode == 'DESISIM':

        # TODO: Not sure about 0 here.
        class_codes[0] = ['']
        class_codes[1] = ['STAR','WD']
        class_codes[2] = ['GALAXY']
        class_codes[3] = ['QSO']

    return class_codes

def get_zconf_codes(mode):

    ## Unobserved is 0
    ## Insufficient confidence is 1
    ## Sufficient confidence is 2

    class_codes = {}

    if mode == 'BOSS':

        class_codes[0] = [0]
        class_codes[1] = [1,2]
        class_codes[2] = [3]

    elif mode == 'DESI':

        # TODO: Not sure about 0 here.
        class_codes[0] = [0]
        class_codes[1] = [1,2,3]
        class_codes[2] = [4]

    elif mode == 'DESISIM':

        # TODO: Not sure about 0 here.
        class_codes[0] = [0]
        class_codes[1] = [1,2,3]
        class_codes[2] = [4]

    return class_codes

def get_desi_targeting_bit_col(period):

    if period == 'survey':
        targeting_bit_col = 'DESI_TARGET'
    elif period == 'sv':
        targeting_bit_col = 'SV1_DESI_TARGET'
    elif period == 'cmx':
        targeting_bit_col = 'CMX_TARGET'
    else:
        raise ValueError('DESI period {} not recognised.'.format(period))

    return targeting_bit_col

# TODO: move this?
absorber_IGM = {
    'Halpha'      : 6562.8,
    'OIII(5008)'  : 5008.24,
    'OIII(4933)'  : 4932.68,
    'Hbeta'       : 4862.68,
    'MgI(2853)'   : 2852.96,
    'MgII(2804)'  : 2803.5324,
    'MgII(2796)'  : 2796.3511,
    'FeII(2600)'  : 2600.1724835,
    'FeII(2587)'  : 2586.6495659,
    'MnII(2577)'  : 2576.877,
    'FeII(2383)'  : 2382.7641781,
    'FeII(2374)'  : 2374.4603294,
    'FeII(2344)'  : 2344.2129601,
    'CIII(1909)'  : 1908.734,
    'AlIII(1863)' : 1862.79113,
    'AlIII(1855)' : 1854.71829,
    'AlII(1671)'  : 1670.7886,
    'FeII(1608)'  : 1608.4511,
    'CIV(1551)'   : 1550.77845,
    'CIV(eff)'    : 1549.06,
    'CIV(1548)'   : 1548.2049,
    'SiII(1527)'  : 1526.70698,
    'SiIV(1403)'  : 1402.77291,
    'SiIV(1394)'  : 1393.76018,
    'CII(1335)'   : 1334.5323,
    'SiII(1304)'  : 1304.3702,
    'OI(1302)'    : 1302.1685,
    'SiII(1260)'  : 1260.4221,
    'NV(1243)'    : 1242.804,
    'NV(1239)'    : 1238.821,
    'LYA'         : 1215.67,
    'SiIII(1207)' : 1206.500,
    'NI(1200)'    : 1200.,
    'SiII(1193)'  : 1193.2897,
    'SiII(1190)'  : 1190.4158,
    'OI(1039)'    : 1039.230,
    'OVI(1038)'   : 1037.613,
    'OVI(1032)'   : 1031.912,
    'LYB'         : 1025.72,
}
