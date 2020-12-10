"""
Weingartner & Draine (2001) dust grain size distributions.
"""

import os
import numpy             as np

from math                import erf
from astropy.io          import ascii

from .  import spectrum
from .. import utils
from .. import constants as c

__all__ = ['WD01']


# Data files for model parameters
MW_caseA = 'Table1_WD2001_Case_A.dat'
MW_caseB = 'Table1_WD2001_Case_B.dat'
LMC_avg  = 'Table3_LMCavg.WD.dat'
LMC_2    = 'Table3_LMC2.WD.dat'
SMC      = 'Table3_SMC.WD.dat'

# Path to tables
DATA_DIR = os.path.join(os.path.dirname(__file__).rstrip('sizedist'), 'tables')


def get_params( Rv=3.1, bc=0., graintype='Graphite', gal='MW', MWcase='A', LMCcase='AVG' ):
    """
    | Get the params from WD01 models, see tables in paper for details
    |
    | Params:
    |        Rv   : float (e.g: 3.1, 4.0, or 5.5)
    |        bc    : float (e.g: 0,1,2,3...), see paper for definition
    |        graintype : string - 'Graphite' or 'Silicate,
    |        gal   : string - 'MW','LMC' or 'SMC'
    |        MWcase : string 'A' or 'B'
    |        LMCcase : string 'AVG' or '2'
    |
    ------------------------------------------
    |
    | Return: (alpha, beta, a_t, a_c, C) : Parameters used in WD01 fits
    |
    """
    if gal == 'MW':
        file_path = os.path.join(DATA_DIR, MW_caseA) if MWcase=='A' else os.path.join(DATA_DIR, MW_caseB)
    elif gal == 'SMC':
        file_path = os.path.join(DATA_DIR, SMC)
    elif gal == 'LMC':
        file_path = os.path.join(DATA_DIR, LMC_avg) if LMCcase=='AVG' else os.path.join(DATA_DIR, LMC_2)
    else:
        print('Error: Gal must be MW, LMC or SMC')
        print('Error: MWcase must be A or B')
        print('Error: LMCcase must be AVG or 2')
        return

    # Read data from table
    try:
        param_table = ascii.read( file_path )
    except:
        print('Error: File %s not found' % (file_path))
        return


    # Get index of rows associated with the input Rv value
    # Rv values are not unique, which is why I can't use a dictionary
    # NOTE: Rv values in param_table['col1'] either a float or '--' (LMC/SMC case)

    if gal == 'MW':
        # np.where returns a tuple of arrays, get the first array
        idx = np.where( (param_table['col1'] == Rv) & (param_table['col2'] == bc) )[0]
    else:
        idx = np.where( (param_table['col2'] == bc) )[0]

    # Index of the row
    if len(idx) == 0:
        print('Error: Rv value not found')
        return
    
    idx = idx[0]

    # Params for the grain type
    if graintype == 'Graphite':
        alpha = param_table['col3'][idx]
        beta  = param_table['col4'][idx]
        a_t   = param_table['col5'][idx]
        a_c   = param_table['col6'][idx]
        C     = param_table['col7'][idx]

    elif graintype == 'Silicate':
        alpha = param_table['col8'][idx]
        beta  = param_table['col9'][idx]
        a_t   = param_table['col10'][idx]
        a_c   = 0.1000
        C     = param_table['col11'][idx]

    else:
        print("Error: Grain-type must be 'Graphite' or 'Silicate'.")
        return

    return (alpha, beta, a_t, a_c, C)





def WD01_size_dist(graintype, a, a_cm, bc, alpha, beta, a_t, a_c, C, npoints, graphitetype='total'):
    """
    | WD01 grain-size distribution of Graphite, see equations 2, 4 and 5 in the paper for details
    | Weingartner_2001_ApJ_548_296
    |
    | Params:
    |        graintype : string - 'Graphite' or 'Silicate' (VSM = Very Small Grain)
    |        graphitetype : string - 'VSG', 'Carbonaceous', 'total'
    |        a         : radii - np.array - grain sizes (um)
    |        a_cm      : radii - np.array - grain sizes (cm)
    |        bc        : float (e.g: 0, 1, 2, 3...)
    |        alpha     : float - fit param
    |        a_t       : float - fit param
    |        C         : float - fit param
    |        npoints   : integer - length of x-axis (radii)
    |
    ------------------------------------------
    |
    | Return: np.array - WD01 grain-size distribution
    |
    """


    # Equations 2 and 4 in the paper
    if graintype == 'Graphite':
        # mc      = 12. * 1.67e-24             # Mass of carbon atom in grams (12 m_p)
        mc      = 1.99236261e-23               # g, Mass of carbon atom in grams (12 m_p)
        rho     = 2.24                         # g cm^-3
        sig     = 0.4
        
        a_01    = 3.5  * c.ANGS2MICRON         # 3.5 angstroms in units of microns
        a_01_cm = a_01 * c.MICRON2CM
        bc1     = 0.75 * bc * 1.e-5
        B_1     = (3./(2.*np.pi)**1.5) * (np.exp(-4.5*sig**2) / (rho*a_01_cm**3 * sig)) * bc1 * mc /\
                  (1. + erf( 3.*sig/np.sqrt(2) + np.log(a_01/(3.5*c.ANGS2MICRON) )/(sig*np.sqrt(2)) ) )

        a_02    = 30.  * c.ANGS2MICRON         # 30 angtroms in units of microns
        a_02_cm = a_02 * c.MICRON2CM
        bc2     = 0.25 * bc * 1.e-5
        B_2     = (3./(2.*np.pi)**1.5) * (np.exp(-4.5*sig**2) / (rho*a_02_cm**3 * sig)) * bc2 * mc /\
                  (1. + erf( 3.*sig/np.sqrt(2) + np.log(a_02/(3.5*c.ANGS2MICRON))/(sig*np.sqrt(2)) ) )

        D       = (B_1/a_cm) * np.exp( -0.5*( np.log(a/a_01)/sig )**2 ) + \
                  (B_2/a_cm) * np.exp( -0.5*( np.log(a/a_02)/sig )**2 )

        id_vsg  = np.where( a < 3.5*c.ANGS2MICRON )
        if np.size(id_vsg) != 0:
            D[id_vsg] = 0.

        
        # Equation 4 in the paper
        fn_graphite  = np.zeros( npoints )
        id1_graphite = np.where( (a > 3.5*c.ANGS2MICRON) & (a < a_t ) )
        id2_graphite = np.where( a >= a_t )

        if np.size(id1_graphite) != 0:
            fn_graphite[id1_graphite] = 1.
        
        if np.size(id2_graphite) != 0:
            fn_graphite[id2_graphite] = np.exp( -( (a[id2_graphite]-a_t) / a_c )**3 )

        if beta >= 0.:
            F_g  = 1. + beta * a / a_t
        if beta < 0.:
            F_g  = 1. / (1. - beta * a / a_t)

        
        if graphitetype == 'VSG':
            return D                                                        # cm^-4 per n_H
        elif graphitetype == 'Carbonaceous':
            return (C/a_cm) * (a/a_t)**alpha * F_g * fn_graphite            # cm^-4 per n_H
        elif graphitetype == 'total':
            return D + (C/a_cm) * (a/a_t)**alpha * F_g * fn_graphite        # cm^-4 per n_H
        else:
            print("Error: Graphite-type must be 'VSG', 'Carbonaceous' or 'total'.")
            return


    
    # Equation 5 in the paper
    if graintype == 'Silicate':
        fn_silicate  = np.zeros( npoints )
        id1_silicate = np.where( (a > 3.5*c.ANGS2MICRON) & (a < a_t ) )
        id2_silicate = np.where( a >= a_t )

        if np.size(id1_silicate) != 0:
            fn_silicate[id1_silicate] = 1.
        if np.size(id2_silicate) != 0:
            fn_silicate[id2_silicate] = np.exp( -( (a[id2_silicate]-a_t)/a_c )**3 )

        F_s = np.zeros( npoints )
        
        if beta >= 0.:
            F_s = 1. + beta * a / a_t
        
        if beta < 0.:
            F_s = 1. / (1. - beta * a / a_t)

        return C/a_cm * (a/a_t)**alpha * F_s * fn_silicate             # cm^-4 per n_H





# DEFAULT_RADII = np.logspace(np.log10(0.005), np.log10(1.0), 50)
DEFAULT_RADII = np.logspace(np.log10(0.0001), np.log10(1.), 200)

def WD01( Rv=3.1, bc=0., radii=DEFAULT_RADII, graintype='Graphite', gal='MW', graphitetype='total', MWcase='A', LMCcase='AVG' ):
    """
    | Get the params from WD01 models, see tables in paper for details
    |
    | Params:
    |        Rv        : float (e.g: 3.1, 4.0, or 5.5)
    |        bc        : float (e.g: 0,1,2,3...)
    |        radii     : np.array - grain sizes (um)
    |        graintype : string - 'Graphite' or 'Silicate
    |        gal       : string - 'MW','LMC' or 'SMC'
    |
    ------------------------------------------
    |
    | Return: sizedist.DustSpectrum object containing a
    |         (grain sizes), nd (dn/da), and md (total mass density of dust)
    |
    """

    if graintype == 'Graphite':
        rho_d = 2.2                              # density of [g cm^-3]
    elif graintype == 'Silicate':
        rho_d = 3.8                              # density of [g cm^-3]
    else:
        print("Error: Grain-type must be 'Graphite' or 'Silicate'.")
        return

    a       = radii                   # [micron]
    a_cm    = radii * c.MICRON2CM
    npoints = np.size( a )

    (alpha, beta, a_t, a_c, C) = get_params( Rv=Rv, bc=bc, graintype=graintype, gal=gal, MWcase=MWcase, LMCcase=LMCcase )

    WD01_sizedist = WD01_size_dist(graintype, a, a_cm, bc, alpha, beta, a_t, a_c, C, npoints, graphitetype=graphitetype)

    mg = (4./3.)* np.pi * a_cm**3 * rho_d             # mass of each dust grain
    Md = utils.xytrapz( a_cm, WD01_sizedist * mg )

    ret     = spectrum.GrainSpectrum()
    ret.a   = a
    ret.rho = rho_d
    ret.nd  = WD01_sizedist * c.MICRON2CM  # cm^-3 per um per n_H
    ret.md  = Md

    return ret