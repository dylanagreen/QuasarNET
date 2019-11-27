from scipy.interpolate import interp1d
from numpy import zeros, arange, array
import numpy as np

def process_preds(preds, lines, lines_bal):
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
        c_line[il], z_line[il] = line_preds_to_properties(preds[il],lines[il])

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
        c_line_bal[il], z_line_bal[il] = line_preds_to_properties(preds[nlines+il],lines_bal[il])

        """
        l = absorber_IGM[lines_bal[il]]
        j = preds[nlines+il][:,:nboxes].argmax(axis=1)
        offset = preds[nlines+il][arange(nspec, dtype=int), nboxes+j]
        c_line_bal[il] = preds[il+nlines][:,:nboxes].max(axis=1)
        z_line_bal[il] = i_to_wave((j+offset)*len(wave)/nboxes)/l-1
        """

    return c_line, z_line, zbest, c_line_bal, z_line_bal

def line_preds_to_properties(line_preds,line):
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
    wave = Wave()
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
