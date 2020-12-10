
import numpy            as np
from scipy.interpolate  import interp1d

from ..            import constants    as c
from ..composition import cmindex      as cmi
from .parse_PAH    import *

__all__ = ['RGscat','Mie','PAH']


class RGscat(object):
    """
    | RAYLEIGH-GANS scattering model.
    | *see* Mauche & Gorenstein (1986), ApJ 302, 371
    | *see* Smith & Dwek (1998), ApJ, 503, 831
    |
    | -- Properties:
    | stype : string : 'RGscat'
    |
    | -- Methods:
    | Qsca( E, a=1.0 [um], cm=cmi.CmDrude() [see cmi.py] )
    |    *returns* scattering efficiency [unitless]
    |
    | scat_angle( a=1.0 [um], E=1.0 [keV] )
    |    *returns* characteristc scattering angle [arcsec keV um]
    |
    | diff( cm=cmi.CmDrude() [see cmi.py], theta=10.0 [arcsec], a=1.0 [um], E=1.0 [keV] )
    |    *returns* differential scattering cross-section [cm^2 ster^-1]
    """

    def __init__(self, stype='RGscat'):
        self.stype = stype

    

    def Qsca( self, E, a=1., cm=cmi.CmDrude() ):

        if np.size(a) != 1:
            print('Error: Radius "a" can only be a single value')
            return

        a_cm    = a * c.MICRON2CM                  # grain radius: cm -- Can only be a single value
        lambda_ = c.HC / E                         # wavelengths:  cm -- Can be many values
        x       = 2. * np.pi * a_cm / lambda_
        m_ci    = cm.rp(E) - 1 + 1j * cm.ip(E)     # the complex index of refraction
        
        return 2. * x**2 * np.abs(m_ci)**2

    

    def scat_angle( self, a=1., E=1. ):    # Characteristic Scattering Angle(s)
        return 1.04 * 60. / (E * a)       # arcsec (keV/E) (um/a)

    

    # Can take multiple theta, but should only use one 'a' value
    # Can take multiple E, but should be same size as theta
    def diff( self, theta, E=1., a=1., cm=cmi.CmDrude() ): # cm^2 ster^-1
        '''
        | Return: 
        |        dsigma/dOmega of values (E0,theta0), (E1,theta1)
        '''

        if np.size(a) != 1:
            print('Error: Radius "a" can only be a single value')
            return
        
        if (np.size(E) > 1) & (np.size(E) != np.size(theta)): # Energy and theta must be of the same size
            print('Error: If more than 01 energy values, energy and theta must have the same size.')
            return

        a_cm    = a * c.micron2cm                   # grain radius: cm -- Can only be a single value
        lambda_ = c.HC / E                          # wavelengths:  cm -- Can be many values
        x       = 2. * np.pi * a_cm / lambda_
        
        energy  = cm.rp(E) - 1 + 1j * cm.ip(E)
        
        thdep   = (2./9.) * np.exp( -(theta/self.scat_angle(a=a, E=E))**2 / 2. )
        dsig    = 2. * a_cm**2 * x**4 * np.abs(energy)**2
        
        return dsig * thdep


'''
Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
to calculate scattering and absorption by a homogenous isotropic
sphere.
From IDL code: code/mie/bhmie_mod.pro
'''

class Mie(object):
    """
    | Mie scattering algorithms of Bohren & Hoffman
    | See their book: *Absorption and Scattering of Light by Small Particles*
    |
    | -- Properties:
    | stype : string : 'Mie'
    |
    | -- Methods:
    | getQs( a=1. [um], E=1. [keV], cm=cmi.CmDrude(), getQ='sca' ['ext','back','gsca','diff'], theta=None [arcsec] )
    |     *returns* Efficiency factors depending on getQ [unitless or ster^-1]
    |
    | Qsca( E [keV], a=1. [um], cm=cmi.CmDrude() )
    |     *returns* Scattering efficiency [unitless]
    |
    | Qext( E [keV], a=1. [um], cm=cmi.CmDrude() )
    |     *returns* Extinction efficiency [unitless]
    |
    | diff( theta [arcsec], E=1. [keV], a=1. [um], cm=cmi.CmDrude() )
    |     *returns* Differential cross-section [cm^2 ster^-1]
    """

    def __init__(self, stype='Mie'):
        self.stype = stype

    def getQs( self, a=1., E=1., cm=cmi.CmDrude(), getQ='sca', theta=None ):  # Takes single a and E argument

        if np.size(a) > 1:
            print('Error: Radius "a" can only be a single value')
            return

        idx_lt90 = np.array([])  # Empty arrays indicate that there are no theta values set
        idx_gt90 = np.array([])  # Do not have to check if theta != None throughout calculation
        s1       = np.array([])
        s2       = np.array([])
        pi       = np.array([])
        pi0      = np.array([])
        pi1      = np.array([])
        tau      = np.array([])
        amu      = np.array([])

        if theta != None:
            if np.size(E) > 1 and np.size(E) != np.size(theta):
                print('Error: If more than 01 energy values, energy and theta must have the same size.')
                return

            if np.size( theta ) == 1:
                theta = np.array( [theta] )       # in degree

            theta_rad = theta * c.ARCS2RAD
            amu       = np.abs( np.cos(theta_rad) )

            idx_lt90  = np.where( theta_rad < np.pi/2. )
            idx_gt90  = np.where( theta_rad >= np.pi/2. )

            theta_size = np.size( theta )
            s1         = np.zeros( theta_size, dtype='complex' )
            s2         = np.zeros( theta_size, dtype='complex' )
            pi         = np.zeros( theta_size, dtype='complex' )
            pi0        = np.zeros( theta_size, dtype='complex' )
            pi1        = np.zeros( theta_size, dtype='complex' ) + 1.
            tau        = np.zeros( theta_size, dtype='complex' )
        # End - if theta != None

        # Complex index of the complex index of refraction
        refrel = cm.rp(E) + 1j*cm.ip(E)

        x      = (2. * np.pi * a*c.MICRON2CM) / (c.HC/E)
        y      = x * refrel
        ymod   = np.abs(y)
        nx     = np.size(x)


        # *** Series expansion terminated after NSTOP terms
        # Logarithmic derivatives calculated from NMX on down

        xstop  = x + 4. * np.cbrt(x) + 2.   # cbrt : cube-root
        test   = np.append( xstop, ymod )
        nmx    = np.max( test ) + 15
        nmx    = np.int32(nmx)

        nstop  = xstop

        d = np.zeros( (nx, nmx+1), dtype='complex' )
        for n in np.arange(nmx-1)+1:  # for n=1, nmx-1 do begin
          en           = nmx - n + 1
          d[:, nmx-n]  = (en/y) - ( 1. / ( d[:,nmx-n+1]+en/y ) )


        # *** Riccati-Bessel functions with real argument X
        # calculated by upward recurrence
        psi0 = np.cos(x)
        psi1 = np.sin(x)
        chi0 = -np.sin(x)
        chi1 = np.cos(x)
        xi1  = psi1 - 1j * chi1

        qsca    = 0.     # scattering efficiency
        gsca    = 0.     # <cos(theta)>
        
        s1_ext  = 0
        s2_ext  = 0
        s1_back = 0
        s2_back = 0
        
        pi_ext  = 0
        pi0_ext = 0
        pi1_ext = 1
        tau_ext = 0
        
        p       = -1.

        for n in range(1, int(np.max(nstop))+1 ):  # for n=1, nstop do begin
            en = n
            fn = (2.*en + 1.)/ (en* (en+1.))

            # for given N, PSI  = psi_n        CHI  = chi_n
            #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
            #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
            # Calculate psi_n and chi_n
            # *** Compute AN and BN:

            #*** Store previous values of AN and BN for use
            #    in computation of g=<cos(theta)>
            if n > 1:
                an1 = an
                bn1 = bn

            if nx > 1:
                ig  = np.where( nstop >= n )

                psi    = np.zeros( nx )
                chi    = np.zeros( nx )

                psi[ig] = (2.*en-1.) * psi1[ig]/x[ig] - psi0[ig]
                chi[ig] = (2.*en-1.) * chi1[ig]/x[ig] - chi0[ig]
                xi      = psi - 1j * chi

                an = np.zeros( nx, dtype='complex' )
                bn = np.zeros( nx, dtype='complex' )

                an[ig] = ( d[ig,n]/refrel[ig] + en/x[ig] ) * psi[ig] - psi1[ig]
                an[ig] = an[ig] / ( ( d[ig,n]/refrel[ig] + en/x[ig] ) * xi[ig] - xi1[ig] )
                bn[ig] = ( refrel[ig]*d[ig,n] + en / x[ig] ) * psi[ig] - psi1[ig]
                bn[ig] = bn[ig] / ( ( refrel[ig]*d[ig,n] + en/x[ig] ) * xi[ig] - xi1[ig] )
            
            else:
            
                psi = (2.*en-1.) * psi1/x - psi0
                chi = (2.*en-1.) * chi1/x - chi0
                xi  = psi - 1j * chi

                an = ( d[0,n]/refrel + en/x ) * psi - psi1
                an = an / ( ( d[0,n]/refrel + en/x ) * xi - xi1 )
                bn = ( refrel*d[0,n] + en / x ) * psi - psi1
                bn = bn / ( ( refrel*d[0,n] + en/x ) * xi - xi1 )
            # End - if


            # *** Augment sums for Qsca and g=<cos(theta)>
            qsca = qsca + ( 2.*en +1. ) * ( np.abs(an)**2 + np.abs(bn)**2 )
            gsca = gsca + ( ( 2.*en+1. ) / ( en*(en+1.) ) ) * ( an.real*bn.real + an.imag*bn.imag )

            if n > 1:
                gsca = gsca + ( (en-1.) * (en+1.)/en ) * \
                                           ( an1.real*an.real + an1.imag*an.imag + bn1.real*bn.real + bn1.imag*bn.imag )

            # *** Now calculate scattering intensity pattern
            #     First do angles from 0 to 90

            # If theta is specified, and np.size(E) > 1,
            # the number of E values must match the number of theta
            # values. 
            pi  = pi1
            tau = en * amu * pi - (en + 1.) * pi0

            if np.size(idx_lt90) != 0:
                antmp = an
                bntmp = bn
                if nx > 1:
                    antmp = an[idx_lt90]
                    bntmp = bn[idx_lt90]  # For case where multiple E and theta are specified

                s1[idx_lt90]  = s1[idx_lt90] + fn* (antmp*pi[idx_lt90]  + bntmp*tau[idx_lt90])
                s2[idx_lt90]  = s2[idx_lt90] + fn* (antmp*tau[idx_lt90] + bntmp*pi[idx_lt90])
            #end - if

            pi_ext = pi1_ext
            tau_ext = en*1.*pi_ext - (en+1.)*pi0_ext

            s1_ext = s1_ext + fn* (an*pi_ext+bn*tau_ext)
            s2_ext = s2_ext + fn* (bn*pi_ext+an*tau_ext)

            # *** Now do angles greater than 90 using PI and TAU from
            #     angles less than 90.
            #     P=1 for N=1,3,...; P=-1 for N=2,4,...

            p = -p

            if np.size(idx_gt90) != 0:
                antmp = an
                bntmp = bn
                if nx > 1:
                    antmp = an[idx_gt90]
                    bntmp = bn[idx_gt90]  # For case where multiple E and theta are specified

                s1[idx_gt90]  = s1[idx_gt90] + fn*p* (antmp*pi[idx_gt90]-bntmp*tau[idx_gt90])
                s2[idx_gt90]  = s2[idx_gt90] + fn*p* (bntmp*pi[idx_gt90]-antmp*tau[idx_gt90])
            #end - if

            s1_back = s1_back + fn*p* (an*pi_ext - bn*tau_ext)
            s2_back = s2_back + fn*p* (bn*pi_ext - an*tau_ext)

            psi0 = psi1
            psi1 = psi
            chi0 = chi1
            chi1 = chi
            xi1  = psi1 - 1j*chi1

            # *** Compute pi_n for next value of n
            #     For each angle J, compute pi_n+1
            #     from PI = pi_n , PI0 = pi_n-1

            pi1  = ( (2.*en + 1.)*amu*pi - (en + 1.)*pi0 ) / en
            pi0  = pi

            pi1_ext = ( (2.*en + 1.)*pi_ext - (en + 1.)*pi0_ext ) / en
            pi0_ext = pi_ext
        # ENDFOR

        # *** Have summed sufficient terms.
        #     Now compute QSCA,QEXT,QBACK,and GSCA
        gsca = 2. * gsca / qsca
        qsca = (2. / x**2) * qsca

        qext  = (4. / x**2) * s1_ext.real
        qback = (np.abs(s1_back)/x)**2 / np.pi

        if getQ == 'sca':
            return qsca
        
        if getQ == 'ext':
            return qext
        
        if getQ == 'back':
            return qback
        
        if getQ == 'gsca':
            return gsca
        
        if getQ == 'diff':
            bad_theta = np.where( theta_rad > np.pi )  # Set to 0 values where theta > !pi
            s1[bad_theta] = 0
            s2[bad_theta] = 0
            return 0.5 * ( np.abs(s1)**2 + np.abs(s2)**2 ) / (np.pi * x**2)
        else:
            return 0.


    def Qsca( self, E, a=1., cm=cmi.CmDrude() ):
        return self.getQs( a=a, E=E, cm=cm )

    def Qext( self, E, a=1., cm=cmi.CmDrude() ):
        return self.getQs( a=a, E=E, cm=cm, getQ='ext' )

    
    def diff( self, theta, E=1., a=1., cm=cmi.CmDrude() ):
        cgeo = np.pi * (a*c.micron2cm)**2

        if np.size(a) != 1:
            print('Error: Radius "a" can only be a single value')
            return
        
        if ( np.size(E) > 1) & (np.size(E) != np.size(theta) ):
            print('Error: If more than 01 energy values, energy and theta must have the same size.')
            return

        dQ  = self.getQs( a=a, E=E, cm=cm, getQ='diff', theta=theta )

        return dQ * cgeo





class PAH( object ):
    """
    | -- Properties:
    | chtype : string : 'ion' or 'neu'
    | stype  : string : 'PAH' + type
    |
    | -- Methods:
    | Qsca( E, a=0.01 [um], cm=None )
    |     *returns* scattering efficiency [unitless]
    |
    | Qabs( E, a=0.01 [um], cm=None )
    |     *returns* absorption efficiency [unitless]
    |
    | Qext( E, a=0.01 [um], cm=None )
    |     *returns* extincton efficiency [unitless]
    """

    def __init__( self, chtype ):
        self.type  = chtype
        self.stype = 'PAH' + chtype

    def get_Q( self, E, qtype, a ):
        try :
            data = parse_PAH( self.type )
        except :
            print('ERROR: Cannot find PAH type', self.type)
            return

        try :
            qvalues  = np.array( data[a][qtype] )
            wvlength = np.array( data[a]['w(micron)'] )
        except :
            print('ERROR: Cannot get grain size', a, 'for', self.stype)
            return

        # Wavelengths were listed in reverse order
        q_interp = interp1d( wvlength[::-1], qvalues[::-1] )

        wavelength = ( c.HC/E ) * 1.e4   # lambda: cm to um
        return q_interp( wavelength )

    def Qabs( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_abs', a )

    def Qext( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_ext', a )

    def Qsca( self, E, a=0.01 ):
        return self.get_Q( E, 'Q_sca', a )