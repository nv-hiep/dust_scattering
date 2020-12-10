
import numpy              as     np
from scipy.interpolate    import interp1d

from .. import constants  as     c
from .. import utils
from .  import scatmodels

from ..composition        import cmindex
from ..                   import sizedist

__all__ = ['ScatModel','DiffScat','SigmaExt','SigmaScat','KappaExt','KappaScat']





def angles( thmin=5., thmax=100., dth=5. ):
    '''
    |
    |Params:
    | thmin=5.0, thmax=100.0, dth=5.0 in [arcsec]
    |
    |Return : np.array - Distribution of angles (theta) [arcsec]
    '''
    return np.arange( thmin, thmax+dth, dth )






#-------------- Tie scattering mechanism to an index of refraction ------------------#

class ScatModel(object):
    """
    | -- Properties:
    | scat_model    : scattering model object - RGscat(), Mie()
    | cmindex_model : cmindex object - CmDrude(), CmGraphite(), CmSilicate()
    | scat_model    : string - 'RGscat', 'Mie'
    | cmtype        : 'Drude', 'Silicate', 'Graphite'
    """
    def __init__( self, scat_model=scatmodels.RGscat(), cmindex_model=cmindex.CmDrude() ):
        self.scat_model    = scat_model
        self.cmindex_model = cmindex_model
        self.scat_type     = scat_model.stype
        self.cmindex_type  = cmindex_model.cmtype
        # cmindex_type choices : 'Drude' (requires rho term only)
        #                  'Graphite' (Carbonaceous grains)
        #                  'Silicate' (Silicate)
        #                  --- Graphite and Silicate values come from Draine (2003)



def create_scat_model( model_name, material_name ):
    '''
    | -- Params:
    | model_name    : string : 'RG' or 'Mie'
    | material_name : string : 'Drude', 'Silicate', 'Graphite', 'SmallGraphite'
    |
    | -- Return:
    | ScatModel object
    '''

    if model_name == 'RG':
        scat_mod = scatmodels.RGscat()
    elif model_name == 'Mie':
        scat_mod = scatmodels.Mie()
    else:
        print('Error: Model name not found')
        return

    if material_name == 'Drude':
        cmi_mod = cmindex.CmDrude()
    
    elif material_name == 'Silicate':
        cmi_mod = cmindex.CmSilicate()
    
    elif material_name == 'Graphite':
        cmi_mod = cmindex.CmGraphite()
    
    elif material_name == 'SmallGraphite':             # Small Graphite ~0.01 micron
        cmi_mod = cmindex.CmGraphite( size='small' )
    
    else:
        print('Error: Complex Index name not found')
        return

    return ScatModel(scat_model=scat_mod, cmindex_model=cmi_mod)








#-------------- Various Types of Scattering Cross-sections -----------------------#

class DiffScat(object):
    """
    | A differential scattering cross-section [cm^2 ster^-1] integrated
    | over dust grain size distribution
    |
    | -- Properties
    | scatm : ScatModel
    | theta : np.array  [arcsec]
    | E     : scalar or np.array : Note, must match number of theta values if size > 1
    | a     : scalar : um
    | dsig  : np.array : cm^2 ster^-1
    """
    def __init__(self, scatm=ScatModel(), theta=angles(), E=1., a=1.):
        self.scatm  = scatm
        self.theta  = theta
        self.E      = E
        self.a      = a

        cm   = scatm.cmindex_model
        scat = scatm.scat_model

        if cm.cmtype == 'Graphite':
            dsig_pe   = scat.Diff(theta=theta, a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='perp'))
            dsig_pa   = scat.Diff(theta=theta, a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='para'))
            self.dsig = (dsig_pa + 2. * dsig_pe) / 3.    # See WD01 paper, Section 2.3
        else:
            self.dsig = scat.Diff(theta=theta, a=a, E=E, cm=cm)






class SigmaScat(object):
    """
    | Total scattering cross-section [cm^2] integrated over a dust grain
    | size distribution
    |
    | -- Properties:
    | scatm : ScatModel
    | E     : scalar or np.array [keV]
    | a     : scalar : um
    | qsca  : scalar or np.array [unitless scattering efficiency]
    | sigma : scalar or np.array [cm^2]
    """
    def __init__(self, scatm=ScatModel(), E=1., a=1.):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        cm   = scatm.cmindex_model
        scat = scatm.scat_model
        # print(cm.citation)

        cgeo = np.pi * (a*c.MICRON2CM)**2

        if cm.cmtype == 'Graphite':
            qsca_pe   = scat.Qsca(a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='perp'))
            qsca_pa   = scat.Qsca(a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='para'))
            self.qsca = (qsca_pa + 2.*qsca_pe) / 3.     # see Weingartner_2001_ApJ_548_296.pdf, Section 2.3
        else:
            self.qsca = scat.Qsca(a=a, E=E, cm=cm)

        self.sigma = self.qsca * cgeo                   # sigma_scat = pi * a**2 * Qscat






class SigmaExt(object):
    """
    | Total EXTINCTION cross-section [cm^2] integrated over a dust grain
    | size distribution
    |
    | -- Properties
    | scatm : ScatModel
    | E     : scalar or np.array [keV]
    | a     : scalar : um
    | qext  : scalar or np.array [unitless extinction efficiency]
    | sigma : scalar or np.array [cm^2]
    """
    def __init__(self, scatm=ScatModel(), E=1., a=1.):
        self.scatm  = scatm
        self.E      = E
        self.a      = a

        if scatm.scat_type == 'RGscat':
            print('Rayleigh-Gans cross-section not currently supported for KappaExt')
            self.qext = None
            self.sigma = None
            return

        cm   = scatm.cmindex_model
        scat = scatm.scat_model
        # print(cm.citation)

        cgeo  = np.pi * np.power(a*c.micron2cm, 2)
        if cm.cmtype == 'Graphite':
            qext_pe = scat.Qext(a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='perp'))
            qext_pa = scat.Qext(a=a, E=E, cm=cmindex.CmGraphite(size=cm.size, orient='para'))
            self.qext = (qext_pa + 2.0*qext_pe) / 3.        # see Weingartner_2001_ApJ_548_296.pdf, Section 2.3  
        else:
            self.qext = scat.Qext(a=a, E=E, cm=cm)
        self.sigma = self.qext * cgeo                       # Sigma_ext = (pi*a**2) * Qext








class KappaScat(object):
    """
    | Opacity to scattering [g^-1 cm^2] integrated over dust grain size distribution.
    |
    | --Properties
    | scatm : ScatModel
    | E     : scalar or np.array : keV
    | dist  : sizedist.DustSpectrum
    | kappa : scalar or np.array : cm^2 g^-1, typically
    """
    def __init__(self, E=1., scatm=ScatModel(), dist=sizedist.MRN77()):
        self.scatm  = scatm
        self.E      = E
        self.dist   = dist

        cm   = scatm.cmindex_model
        scat = scatm.scat_model
        # print(cm.citation)

        cgeo = np.pi * (dist.a * c.MICRON2CM)**2

        qsca    = np.zeros(shape = (np.size(E), np.size(dist.a)))
        qsca_pe = np.zeros(shape = (np.size(E), np.size(dist.a)))
        qsca_pa = np.zeros(shape = (np.size(E), np.size(dist.a)))

        # Test for graphite case
        if cm.cmtype == 'Graphite':
            cmGraphitePerp = cmindex.CmGraphite(size=cm.size, orient='perp')
            cmGraphitePara = cmindex.CmGraphite(size=cm.size, orient='para')

            if np.size(dist.a) > 1:
                for i in range(np.size(dist.a)):
                    qsca_pe[:,i] = scat.Qsca(E, a=dist.a[i], cm=cmGraphitePerp)
                    qsca_pa[:,i] = scat.Qsca(E, a=dist.a[i], cm=cmGraphitePara)
            else:
                qsca_pe = scat.Qsca(E, a=dist.a, cm=cmGraphitePerp)
                qsca_pa = scat.Qsca(E, a=dist.a, cm=cmGraphitePara)

            qsca = (qsca_pa + 2. * qsca_pe) / 3.              # see Weingartner_2001_ApJ_548_296.pdf, Section 2.3  

        else:
            if np.size(dist.a) > 1:
                for i in range(np.size(dist.a)):
                    qsca[:,i] = scat.Qsca(E, a=dist.a[i], cm=cm)
            else:
                qsca = scat.Qsca(E, a=dist.a, cm=cm)

        if np.size(dist.a) == 1:
            kappa = dist.nd * qsca * cgeo / dist.md
        elif np.all(dist.nd == 0.):
            kappa = np.array( [0.]*np.size(E) )
        else:
            kappa = np.array([])
            for j in range(np.size(E)):
                kappa = np.append(kappa, utils.xytrapz(dist.a, dist.nd * qsca[j,:] * cgeo) / dist.md)

        self.kappa = kappa







class KappaExt(object):
    """
    | Opacity to EXTINCTION [g^-1 cm^2] integrated over dust grain size
    | distribution
    |
    | -- Properties
    | scatm : ScatModel
    | E     : scalar or np.array [keV]
    | dist  : sizedist.DustSpectrum
    | kappa : scalar or np.array [cm^2 g^-1, typically]
    """
    def __init__(self, E=1., scatm=ScatModel(), dist=sizedist.MRN77()):
        self.scatm  = scatm
        self.E      = E
        self.dist   = dist

        if scatm.scat_type == 'RGscat':
            print('Rayleigh-Gans cross-section not currently supported for KappaExt')
            self.kappa = None
            return

        cm   = scatm.cmindex_model
        scat = scatm.scat_model
        # print(cm.citation)

        cgeo = np.pi * (dist.a * c.MICRON2CM)**2

        qext    = np.zeros(shape=(np.size(E), np.size(dist.a)))
        qext_pe = np.zeros(shape=(np.size(E), np.size(dist.a)))
        qext_pa = np.zeros(shape=(np.size(E), np.size(dist.a)))

        # For graphite case
        # See https://iopscience.iop.org/article/10.1086/318651/pdf Section 2.3
        # Qext = [Qext_epsilon_paralell + 2*Qext_epsilon_perpendicular] / 3.0
        if cm.cmtype == 'Graphite':
            cmGraphitePerp = cmindex.CmGraphite(size=cm.size, orient='perp')
            cmGraphitePara = cmindex.CmGraphite(size=cm.size, orient='para')

            if np.size(dist.a) > 1:
                for i in range(np.size(dist.a)):
                    qext_pe[:,i] = scat.Qext(E, a=dist.a[i], cm=cmGraphitePerp)
                    qext_pa[:,i] = scat.Qext(E, a=dist.a[i], cm=cmGraphitePara)
            else:
                qext_pe = scat.Qext(E, a=dist.a, cm=cmGraphitePerp)
                qext_pa = scat.Qext(E, a=dist.a, cm=cmGraphitePara)

            qext    = (qext_pa + 2. * qext_pe) / 3.

        else:
            if np.size(dist.a) > 1:
                for i in range(np.size(dist.a)):
                    qext[:,i] = scat.Qext(E, a=dist.a[i], cm=cm)
            else:
                qext = scat.Qext(E, a=dist.a, cm=cm)

        if np.size(dist.a) == 1:
            kappa = dist.nd * qext * cgeo / dist.md
        elif np.all(dist.nd == 0.):
            kappa = np.array( [0.]*np.size(E) )
        else:
            kappa = np.array([])
            for j in range(np.size(E)):
                kappa = np.append(kappa, utils.xytrapz(dist.a, dist.nd * qext[j,:] * cgeo) / dist.md)

        self.kappa = kappa