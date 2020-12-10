import numpy             as np
from .. import constants as c
from .. import utils

__all__ = ['GrainSpectrum']

# Default values
MDUST    = 1.5e-5  # g cm^-2 (dust mass column)
RHO      = 3.0     # g cm^-3 (average grain material density)

NPOINTS  = 100     # default number for grain size dist resolution
PDIST    = 3.5     # default slope for power law distribution

# min and max grain radii for MRN distribution
AMIN     = 0.005   # micron
AMAX     = 0.3     # micron


def xaxis_radii(amin=AMIN, amax=AMAX, npoints=NPOINTS, log=False):
    '''
    Make default xaxis of grain-sizes
    '''
    return np.logspace(np.log10(amin), np.log10(amax), npoints) if log else np.linspace(amin, amax, npoints)



RADII = xaxis_radii()


def num_dens(a=RADII, rho=RHO, p=PDIST, md=MDUST):
    ap    = a**(-p)                         # um^-p
    gdens = (4./3.) * np.pi * rho
    dmda  = ap * gdens * (a * c.MICRON2CM)**3    # g um^-p
    const = md / utils.xytrapz(a, dmda)          # trapz(y,x); cm^-? um^p-1
    return const * ap


NUM_DENS = num_dens()



class GrainSpectrum(object):  # radius (a), number density (nd), and mass density (md)
    """
    | -- Properties
    | dist : A dust distribution that contains attributes a and rho and ndens function
    | md   : mass density of dust [units arbitrary, usually g cm^-2]
    | nd   : number density of dust [set by md units]
    | rho  : dust grain material density [g cm^-3]
    |
    """
    def __init__(self, a=RADII, rho=RHO, p=PDIST, nd=NUM_DENS, md=MDUST):
        self.md  = md
        self.a   = a
        self.p   = p
        self.rho = rho
        self.nd  = nd


    def calc_attr(self, amin=AMIN, amax=AMAX, p=PDIST, rho=RHO, md=MDUST, npoints=NPOINTS, log=False):
        self.a   = xaxis_radii(amin=amin, amax=amax, npoints=npoints, log=log)
        self.p   = p
        self.md  = md
        self.rho = rho
        self.nd  = num_dens(a=self.a, rho=rho, p=p, md=md)


    def mass_column(self):
        '''
        Purpose : Calculate the total mass column of dust (g cm^-2)
        |
        |Return:
        |       total mass column of dust (g cm^-2)
        '''
        mass = (4.*np.pi/3.) * self.rho * (c.MICRON2CM*self.a)**3
        
        if np.size(self.a) == 1:
            return mass*self.nd
        
        return utils.xytrapz(self.a, mass*self.nd)