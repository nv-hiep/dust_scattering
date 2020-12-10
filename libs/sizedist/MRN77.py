import numpy as np
from .. import constants as c
from . import spectrum

__all__ = ['MRN77']

# Some default values
MDUST    = 1.5e-5  # g cm^-2 (dust mass column)
RHO      = 3.      # g cm^-3 (average grain material density)

PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron

#-----------------------------------------------------------------

def MRN77(amin=AMIN, amax=AMAX, p=PDIST, rho=RHO, md=MDUST, npoints=100, log=False):
    """
    | Returns a dust spectrum for a power law distribution of dust grains
    |
    | **INPUTS**
    | amin : [micron]
    | amax : [micron]
    | p    : scalar for dust power law dn/da proptional to a^-p
    | rho  : grain density [g cm^-3]
    | md   : mass density [g cm^-2 or g cm^-3]
    | **kwargs : See libs.sizedist.PowerLaw keywords
    |
    | **RETURNS**
    |            GrainSpectrum object
    """
    ret = spectrum.GrainSpectrum()
    ret.calc_attr(amin=amin, amax=amax, p=p, rho=rho, md=md, npoints=npoints, log=log)
    
    return ret