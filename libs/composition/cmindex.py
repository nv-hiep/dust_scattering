
import os
import numpy           as np
from scipy.interpolate import interp1d

from .. import utils
from .. import constants as c


#--- Complex index of refraction ---#

def cmfile_path( name ):
    return os.path.join(os.path.join(os.path.dirname(__file__).rstrip('composition'), 'tables'), name)



class CmDrude(object):
    '''
    | Formula for calculating the complex index of refraction [E in keV] from Drude approximation
    |
    | -- Properties:
    | cmtype   : string - 'Drude'
    | rho      : float - grain density [g cm^-3]
    | citation : A string containing citation to original work
    |
    | -- Methods:
    | rp(E)  : real part of complex index of refraction [E in keV]
    | ip(E)  : imaginary part of complex index of refraction [always 0.0]
    '''
    def __init__(self, rho=3.):  # Returns a CM using the Drude approximation
        self.cmtype   = 'Drude'
        self.rho      = rho
        self.citation = "Using the Drude approximation.\nBohren, C. F. & Huffman, D. R., 1983, Absorption and Scattering of Light by Small Particles (New York: Wiley)"

    # Real part
    def rp(self, E):
        return 1. + self.rho / (2.*c.M_p)*c.R_e / (2.*np.pi) * (c.HC/E)**2

    # Imaginary part, all zeros
    def ip(self, E):
        if np.size(E) > 1:
            return np.zeros(np.size(E))
        return 0.




class CmGraphite(object):
    '''
    | Get the complex index of refraction [E in keV] from Draine2003
    |
    | -- Properties
    | cmtype : 'Graphite'
    | size   : 'big' or 'small'
    | orient : 'perp' or 'para'
    | citation : A string containing citation to original work
    |
    | To calculate real and imaginary parts of the complex index of refraction:
    | rp(E)  : scipy.interp1d object
    | ip(E)  : scipy.interp1d object [E in keV]
    '''
    def __init__(self, size='big', orient='perp'):
        '''
        | size : string ('big' or 'small')
        |     : 'big' gives results for 0.1 um sized graphite grains at 20 K [Draine (2003)]
        |     : 'small' gives results for 0.01 um sized grains at 20 K
        | orient : string ('perp' or 'para')
        |       : 'perp' gives results for E-field perpendicular to c-axis
        |       : 'para' gives results for E-field parallel to c-axis
        '''
        
        self.cmtype   = 'Graphite'
        self.size     = size
        self.orient   = orient
        self.citation = "Using optical constants for graphite,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"

        # From Draine, B. T. 2003
        res = utils.restore( cmfile_path('Complex_index_Draine2003.pysav') )  # read in index values

        '''
        Dict 'res' with keys:
        [
        'Cpa_001_lam', 'Cpe_001_lam', 'Cpa_001_re', 'Sil_im', 'Cpe_001_im', 'Cpe_010_re', 
        'Cpe_001_re', 'Sil_lam', 'Cpa_010_im', 'Cpe_010_im', 'Cpa_010_lam', 'Cpa_001_im',
        'Cpa_010_re', 'Cpe_010_lam', 'Sil_re'
        ]

        | C -> Graphite
        | Sil -> Silicate
        |
        | pe -> perpendicular
        | pa -> parallel
        | 
        | re -> real part
        | im -> imaginary part
        | 
        | lam -> lambda (wavelength)
        |
        | 010 -> big grains
        | 001 -> small grains
        |
        '''
        keystring = 'C'
        keystring += 'pe_'  if orient == 'perp' else 'pa_'
        keystring += '010_' if size == 'big'    else '001_'

        wavelength = res[keystring + 'lam']
        real       = res[keystring + 're']
        imaginary  = res[keystring + 'im']

        energy = c.HC / c.MICRON2CM / wavelength # keV
        self.rp  = interp1d( energy, real )
        self.ip  = interp1d( energy, imaginary )







class CmSilicate(object):
    '''
    | Get the complex index of refraction [E in keV] from Draine2003
    |
    | -- Properties
    | cmtype : 'Silicate'
    | size   : 'big' or 'small'
    | orient : 'perp' or 'para'
    | citation : A string containing citation to original work
    |
    | To calculate real and imaginary parts of the complex index of refraction:
    | rp(E)  : scipy.interp1d object
    | ip(E)  : scipy.interp1d object [E in keV]
    '''
    def __init__( self ):
        self.cmtype = 'Silicate'
        self.citation = "Using optical constants for astrosilicate,\nDraine, B. T. 2003, ApJ, 598, 1026\nhttp://adsabs.harvard.edu/abs/2003ApJ...598.1026D"


        res = utils.restore( cmfile_path('Complex_index_Draine2003.pysav') )
        '''
        Dict 'res' with keys:
        [
        'Cpa_001_lam', 'Cpe_001_lam', 'Cpa_001_re', 'Sil_im', 'Cpe_001_im', 'Cpe_010_re', 
        'Cpe_001_re', 'Sil_lam', 'Cpa_010_im', 'Cpe_010_im', 'Cpa_010_lam', 'Cpa_001_im',
        'Cpa_010_re', 'Cpe_010_lam', 'Sil_re'
        ]

        | C -> Graphite
        | Sil -> Silicate
        |
        | pe -> perpendicular
        | pa -> parallel
        | 
        | re -> real part
        | im -> imaginary part
        | 
        | lam -> lambda (wavelength)
        |
        | 010 -> big grains
        | 001 -> small grains
        |
        '''

        wavelength = res['Sil_lam']
        real       = res['Sil_re']
        imaginary  = res['Sil_im']

        energy   = c.HC / c.MICRON2CM / wavelength # keV
        self.rp  = interp1d( energy, real )
        self.ip  = interp1d( energy, imaginary )









#------------- A quick way to grab a single CM ------------

def getCM( E, model=CmDrude() ):
    """
    | -- params
    | E     : scalar or np.array [keV]
    | model : any Cm-type object
    |
    | -- return
    | Complex index of refraction : scalar or np.array of dtype='complex'
    """
    return model.rp(E) + 1j * model.ip(E)
