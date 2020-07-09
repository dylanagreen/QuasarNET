from scipy.interpolate import interp1d
from numpy import zeros, arange, array
import numpy as np

def process_preds(preds, lines, lines_bal, wave=None, model_type='boxes'):
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
        c_line[il], z_line[il] = line_preds_to_properties(preds[il],lines[il],wave=wave,model_type=model_type)

    # Get the best estimate of z from the most confident line
    zbest = z_line[c_line.argmax(axis=0),arange(nspec)]
    zbest = array(zbest)

    nlines_bal = len(lines_bal)
    c_line_bal=zeros((nlines_bal, nspec))
    z_line_bal=zeros((nlines_bal, nspec))

    for il in range(len(lines_bal)):
        c_line_bal[il], z_line_bal[il] = line_preds_to_properties(preds[nlines+il],lines_bal[il],wave=wave,model_type=model_type)

    return c_line, z_line, zbest, c_line_bal, z_line_bal

def line_preds_to_properties(line_preds,line,wave=None, model_type='boxes'):
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

    # Construct an interpolator to go from the index along a wave vector to the
    # wavelength associated with this position.
    if wave is None:
        print('WARN: No wave grid information provided; using default:')
        wave = Wave()
        print(' - lmin={}, lmax={}, dll={}'.format(10**wave.llmin,10**wave.llmax,wave.dll))
    i_to_wave = interp1d(arange(len(wave.wave_grid)), wave.wave_grid,
            bounds_error=False, fill_value='extrapolate')

    if model_type == 'boxes':
        nspec, nboxes = line_preds.shape
        nboxes //= 2

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

    elif model_type == 'noboxes':

        #######################################################################
        ### JAF: This is the default. However, it means we can only predict a
        ### redshift that corresponds to a point in our wavelength grid.
        #######################################################################

        """
        l = absorber_IGM[line]
        j = line_preds.argmax(axis=1)
        c_line = line_preds.max(axis=1)
        z_line = i_to_wave(j)/l-1
        """

        #######################################################################
        ### JAF: Alternatively, we can use the predicted values in all cells to
        ### compute a mean (need to use wavelength grid cell widths to ensure
        ### no bias).
        #######################################################################

        """
        # For the moment, use the same method of determining confidence.
        l = absorber_IGM[line]
        c_line = line_preds.max(axis=1)

        # Determine redshift by
        wave_edges = np.concatenate([[wave.wave_grid[0]-(wave.wave_grid[1]-wave.wave_grid[0])/2], (wave.wave_grid[1:]+wave.wave_grid[:-1])/2., [wave.wave_grid[-1]+(wave.wave_grid[-1]-wave.wave_grid[-2])/2]])
        wave_widths = wave_edges[1:] - wave_edges[:-1]
        z_line = ((line_preds*wave.wave_grid[None,:]*wave_widths[None,:]).sum(axis=1)/(line_preds*wave_widths[None,:]).sum(axis=1))/l - 1
        """
        
        #######################################################################
        ### JAF: Alternatively, we can curve fit the pseudo-pdf.
        #######################################################################

        from scipy.optimize import curve_fit

        # For the moment, use the same method of determining confidence.
        l = absorber_IGM[line]
        c_line = line_preds.max(axis=1)

        line_width = 50.
        def gaussian_pseudo_pdf(x, *p):
            mu = np.array(p)
            g = np.exp(-(x-mu[:,None])**2/(line_width**2))
            return g.flatten()

        j = line_preds.argmax(axis=1)
        p0 = i_to_wave(j)/l-1

        wave_line = np.zeros(c_line.shape)
        for i in range(len(z_line)):
            coeff, var_matrix = curve_fit(gaussian_pseudo_pdf, wave.wave_grid, box_line[i,:].flatten(), p0=p0[i])
            test_estimates += [coeff]

        z_line = wave_line/l - 1

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

    elif (mode == 'DESI') or (mode == 'DESI_SPECTRA') or (mode == 'DESI_SIM'):
        tid_field['TARGETID'] = 'TARGETID'

    elif mode == 'DESI_COADD':
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

    elif (mode == 'DESI') or (mode == 'DESI_SPECTRA') or (mode == 'DESI_SIM'):
        spid_fields['SPID0'] = 'TILEID'
        spid_fields['SPID1'] = 'NIGHT'
        spid_fields['SPID2'] = 'FIBER'

    elif mode == 'DESI_COADD':
        spid_fields['SPID0'] = 'BRICKNAME'
        spid_fields['SPID1'] = 'BRICK_OBJID'
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

    elif (mode == 'DESI') or (mode == 'DESI_SPECTRA') or (mode == 'DESI_SIM'):
        truth_fields['Z'] =         'Z_VI'
        truth_fields['OBJCLASS'] =  'CLASS_PERSON'
        truth_fields['Z_CONF'] =    'Z_CONF_PERSON'

    elif mode == 'DESI_SIM':

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

    elif mode == (mode == 'DESI') or (mode == 'DESI_SPECTRA') or (mode == 'DESI_SIM'):
        bal_fields['BAL_FLAG'] = 'BAL_FLAG_VI'
        bal_fields['BI_CIV'] =   'BI_CIV'

    elif mode == 'DESI_SIM':
        bal_fields['BAL_FLAG'] = 'BAL_FLAG_VI'
        bal_fields['BI_CIV'] =   'BI_CIV'

    elif mode == None:
        bal_fields['BAL_FLAG'] =  ''
        bal_fields['BI_CIV'] =    ''

    return bal_fields

def get_targeting_bits(mode,verbose=True,desi_period='survey',desi_cmx_bitname=None):

    if mode =='BOSS':
        tb = {'BOSS_TARGET1': [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44],
              'EBOSS_TARGET0': [10,11,12,13,14,15,16,17,18,20,22,30,31,33,34,35,40],
              'EBOSS_TARGET1': [9,10,11,12,13,14,15,16,17,18,30,31],
              'EBOSS_TARGET2': [0,2,4,20,21,23,24,25,26,27,31,32,33,34,50,51,
                52,53,54,55,56,57,58,59,60,61,62],
              'ANCILLARY_TARGET1': [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,
                23,24,25,26,27,28,29,30,31,50,51,52,53,54,55,58,59],
              'ANCILLARY_TARGET2': [0,1,2,3,4,5,7,8,9,10,13,14,15,24,
                25,26,27,31,32,33,53,54,55,56]
              }

    elif 'DESI' in mode:
        try:
            import desitarget
            if desi_period=='survey':
                b = np.log2(desitarget.targetmask.desi_mask.mask('QSO')).astype('int')
                tb = {'DESI_TARGET': [b],
                      }
            elif desi_period=='sv':
                b = np.log2(desitarget.sv1.sv1_targetmask.desi_mask.mask('QSO')).astype('int')
                tb = {'SV1_DESI_TARGET': [b],
                      }
            elif desi_period=='cmx':
                b = np.log2(desitarget.targetmask.desi_mask.mask(desi_cmx_bitname)).astype('int')
                tb = {'CMX_TARGET': [b],
                      }

        except ImportError:
            if verbose:
                print("WARN: can't load desi_mask, using hardcoded targeting value!")
            if desi_period=='survey':
                tb = {'DESI_TARGET': [2],
                      }
            elif desi_period=='sv':
                tb = {'SV1_DESI_TARGET': [2],
                      }
            elif desi_period=='cmx':
                if desi_cmx_bitname=='MINI_SV_QSO':
                    b = 55
                elif desi_cmx_bitname=='SV0_QSO':
                    b = 12
                else:
                    print('ERROR: cmx bit name {} not found, using SV0_QSO'.format(cmx_bitname))
                    b = 12
                tb = {'CMX_TARGET': [b],
                      }

    return tb

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
