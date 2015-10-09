import scipy
from scipy import signal
import pylab
import atpy
from SuzPyUtils import filter
#from corot import defs as corotdefs
from SuzPyUtils.multiplot import *

RSUN = 6.96e8
DAY2S = 86400.0

def simrv(time, flux, strad = 1.0, nmed = 11, nlin = 5, \
              kvc = 0.0, phi0 = None, f = None, doplot = False):
    '''Simulate RV time series given photometric time series.  Assumes
    input flux array has already been normalised & smoothed, but does
    very light smoothing (nmed=nbin=5 by default)'''
    ln = scipy.isnan(flux)
    npt = len(time)
    phi = filter.filt1d(flux, 1, nlin)
#    phi = signal.medfilt(flux, nmed) 
#    phi = scipy.copy(flux)
    if doplot == True:
        ee = dofig(1, 1, 3)
        ax1 = doaxes(ee, 1, 3, 0, 0)
        pylab.plot(time, flux, 'k.')
        pylab.plot(time, phi, 'r-')
        pylab.ylabel('Norm. flux')
    dt = (time[1:] - time[:npt-1])
    dts = dt * DAY2S
    tout = time[:npt-1] + 0.5 * dt
    phidot = (phi[1:] - phi[:npt-1]) / dts
    if doplot == True:
        ax2 = doaxes(ee, 1, 3, 0, 1, sharex = ax1)
        pylab.plot(tout, phidot, 'k.')
        pylab.ylabel('Flux. der.')
    phidot = filter.filt1d(phidot, 1, nlin)
    if doplot == True:
        pylab.plot(tout, phidot, 'r-')
    lf = scipy.isfinite(phi)
    phimin = min(phi[lf])
    phimax = max(phi[lf])
    med, sig = filter.medsig(phi)
    if phi0 == None:
        phi0 = phimax + sig
    if f == None:
        f = (phi0 - phimin) / phi0
    print phimin, phimax, sig, phi0, f
    rstar = strad * RSUN
    print rstar / f
    if f < 1e-6:
        rvout = tout - tout
    else:
        rvout = - phidot/phi0 * (1 - phi[:npt-1] / phi0) * rstar / f
        if kvc != 0:
            rvout += (1 - phi[:npt-1] / phi0)**2 * kvc / f
    rvout[ln[0:npt-1]] = scipy.nan
    if doplot == True:
        ax3 = doaxes(ee, 1, 3, 0, 2, sharex = ax1)
        pylab.plot(tout, 1 - phi[:npt-1] / phi0)
        # pylab.plot(tout, rvout, 'r-')
        pylab.ylabel('RV (m/s)')
        pylab.xlabel('time (days)')
        pylab.xlim(time.min(), time.max())
    return tout, rvout

def rv_sample(obs = None, tspan = 180, npernight = 3, drun = 10, \
                  nrun = 3, nrand = 10, dnight = 8./24.):
    if obs != None:
# Read in RV data 
        if obs == 'corot7':
            rv = atpy.Table(corotdefs.ROOTDIR + 'LRa01/cands/corot7_rv.ipac')
            time = rv.JDB
        if obs == 'hd189':
            rv = atpy.Table('/Users/suz/Data/HD189_rv.ipac')
            time = rv.hjd
    else:
# One point per night
        days = scipy.arange(tspan)
        dt_night = dnight / float(npernight+1)
# Multiple points per night, with small deviations from regularity
        obs = scipy.zeros((tspan, npernight)) 
        for i in scipy.arange(npernight):
            obs[:,i] = days[:] + dt_night * float(i) + \
                pylab.normal(0, dt_night/2., tspan)
# Select points in "intensive" runs
        if drun == tspan:
            take = scipy.ones((tspan, npernight), 'int')
        else:
            take = scipy.zeros((tspan, npernight), 'int')
            for i in scipy.arange(nrun):
                ok = 0
                while ok == 0:
                    tstart = scipy.fix(scipy.rand(1) * float(tspan))
                    tstart = tstart[0]
                    tend = tstart + drun
                    if tend > tspan: continue
                    if take[tstart:tend,:].any(): continue
                    take[tstart:tend,:] = 1
                    ok = 1
# Select additional individual points
        ntot = tspan*npernight
        obs = scipy.reshape(obs, ntot)
        take = scipy.reshape(take, ntot)
        index = scipy.argsort(obs)
        obs = obs[index]
        take = take[index]
        for i in scipy.arange(nrand):
            ok = 0
            while ok == 0:
                t = scipy.fix(scipy.rand(1) * float(ntot))
                t = t[0]
                if take[t] == 1: continue
                take[t] = 1
                ok = 1
        time = obs[(take==1)]
    time -= time[0]
    return time

def rv_noise(time, tau = 0.5, ninit = 10):
    npt = len(time)
    # Burn-in section
    dt = 2.3 * tau / float(ninit)
    tinit = (scipy.arange(ninit) + 1) * dt
    tsim = scipy.append(time[0] - tinit[::-1], time)
    nsim = len(tsim)
    # Instead of Gaussian white noise, to simulate effects of moon and
    # such like, each point in the "WGN" component is drawn from a
    # distribution with a different sigma each night, where the sigma itself is
    # drawn from a powerlaw distribution between 1 and 10
    wt = pylab.normal(0, 1, nsim)
    ye = scipy.ones(nsim)
    nights = scipy.fix(tsim)
    unights = scipy.unique(nights)
    nnights = len(unights)
    sigma = 10.0 ** (scipy.rand(nnights))**4
    for i in scipy.arange(nnights):
        l = scipy.where(nights == unights[i])[0]
        if l.any():
            wt[l] *= sigma[i]
            ye[l] = sigma[i]
    # Now apply MA model with exponentially decaying correlation
    ysim = scipy.copy(wt)
    for i in scipy.arange(nsim):
        j = i - 1
        coeff = 1.0
        while (j > 0) * (coeff > 0.1):
            dt = tsim[i] - tsim[j]
            coeff = scipy.exp(-dt/tau)
            ysim[i] += coeff * wt[j]
            j -= 1
    # Discard burn-in and globally re-scale before returning
    yout = ysim[ninit:]
    eout = ye[ninit:]
    med, sig = filter.medsig(yout)
    yout /= sig
    eout /= sig
    return yout, eout

# def sun_tests():
#     X = scipy.genfromtxt('/Volumes/Data2/Sun/meunier/low.dat', skip_header=1)
#     time = X.T[0]
#     flux = X.T[1]
#     rv = X.T[3]
#     med, sig = filter.medsig(flux)
#     flux /= med
#     res = simrv(time, flux, strad = 1.0, nmed = 3, nlin = 1, \
#                     kvc = 2000.0, doplot = True)
#     pylab.plot(time, rv, 'g-')
#     return
