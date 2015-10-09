import numpy as np
import pylab as pl
from spot_model import *
from plots import *

DEG2RAD = np.pi / 180.0

def singlespot():
    '''
    Plot time series and periodogram for 4 different versions of a single spot:
    1. equatorial, no limb darkening, no convective
       blue-shift suppression
    2. equatorial with limb darkening, no convective
       blue-shift suppression
    3. high latitude, high inclination, with limb darkening,
       no convective blue-shift suppression
    4. high latitude, high inclination, with limb darkening,
       Sun-like convective blue-shift suppression
    The time-series are plotted as a function of phase.
    '''
    pars = spots(4)
    pars.u[1] = 0.6 # add limb-darkening
    pars.lat[2:] = 60.0 * DEG2RAD # non-equatorial spot 
    pars.incl[2:] = 70.0 * DEG2RAD # on inclined star
    pars.vconv[:3] = 0 # no convective blue-shift effect
    time, dF, dRV, bis = genTSreg(pars)
    plotTSPer(time, dF, dRV, bis, period = pars.period[0], fmp = 6, \
                  figno = [1,2], xper = True)
    return

def singlespot_evol():
    '''
    Plot time series and periodogram for a single spot that evolves.
    The time-series is plotted as a function of time.
    '''
    pars = spots(1)
    pars.incl[0] = 60.0 * DEG2RAD
    pars.lat[0] = 70.0 * DEG2RAD
    pars.decay[0] = 4. * pars.period[0]
    pars.pk[0] = 10. * pars.period[0]
    time, dF, dRV, bis = genTSreg(pars)
    plotTSPer(time, dF, dRV, bis, period = pars.period[0], fmp = 6, \
                  figno = [1,2])

def twospot():
    '''
    Two 2-spot examples, both with evolution, but one also hase differential rotation
    '''
    pars = spots(2)
    pars.incl[:] = 60.0 * DEG2RAD
    pars.lat[:] = 70.0 * DEG2RAD
    pars.decay[:] = 4. * pars.period[:]
    pars.pk[0] = 8. * pars.period[0]
    pars.pk[1] = 16. * pars.period[1]
    pars.phase[1] = 0
    per = pars.period[0]
    time, dF1, dRV1, bis1 = genTSreg(pars, sum = True)
    pars.period[0] = 4.75
    pars.period[1] = 5.25    
    time, dF2, dRV2, bis2 = genTSreg(pars, sum = True)
    nobs = len(time)
    dF = np.array([dF1, dF2]).reshape((2, nobs))
    dRV = np.array([dRV1, dRV2]).reshape((2, nobs))
    bis = np.array([bis1, bis2]).reshape((2, nobs))
    plotTSPer(time, dF, dRV, bis, period = per, fmp = 4, \
                  figno = [1,2])
    return

def manyspots(diffrot = None):
    '''
    Examples with different numbers of spots ranging from 2 to 200,
    and random distributions for the spot parameters
    '''
    Nspots = [5, 20, 200]
    nex = len(Nspots)
    nper = 24
    npper = 1000
    for i in np.arange(nex):
        pars = spots(Nspots[i])
        pars.lat = np.arcsin(np.random.uniform(0,1,Nspots[i]))
        if diffrot:
            omega_eq = 2 * np.pi / pars.period
            omega = omega_eq * (1 - diffrot  * np.sin(pars.lat)**2)
            pars.period[i] = 2 * np.pi / omega
        per_mean = pars.period.mean()
        pars.incl[:] = 60.0 * DEG2RAD
        pars.amax = 10.0**np.random.uniform(-3,-2, Nspots[i]) / float(Nspots[i])
        pars.decay = pars.period * 10.0**(np.random.normal(0, 0.5, Nspots[i]))
        pars.phase = np.random.uniform(0, 2 * np.pi, Nspots[i])
        pars.pk = np.random.uniform(2 * per_mean, (nper + 4) * per_mean, Nspots[i])
        time, dF, dRV, bis = genTSreg(pars, sum = True, nper = nper, npper = npper)
        plotTSPer(time, dF, dRV, bis, period = per_mean, fmp = 6, \
                  figno = [1,2])
        print '%d spots' % Nspots[i]
        raw_input()
    return
