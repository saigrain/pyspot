import numpy as np
import pylab as pl
from gls import sinefitm
from multiplot import dofig, doaxes

fac1 = 100
fac2 = 1
fac3 = 1
ls = ['-','--',':','-.']
mrk = ['.',',','+','x']
col = ['k','c','m','y']

def plotTS(time, y1, y2, y3 = None, figno = 1, discrete = True, \
                  savefile = None, period = None, xper = False):
    '''Plot light and RV curve(s)'''
    M, N = np.shape(y1)
    if discrete == True:
        m1 = np.copy(mrk)
    else:
        m1 = np.copy(ls)
    if (xper == True) * (period != None):
        tt = time / period - 0.5
        xr = [-0.5,0.5]
        xttl = 'phase'
    else:
        tt = time
        xr = np.nanmin(time), np.nanmax(time)
        xttl = 'time (days)'
    if y3 == None:
        ny = 2
    else:
        ny = 3
    ee = dofig(figno, 1, ny, aspect = 1)
    ax1 = doaxes(ee, 1, ny, 0, 0)
    for i in np.arange(M):
        pl.plot(tt, y1[i,:] * fac1, m1[i], c = col[i])
    pl.ylabel(r"$\Delta F$ (\%)")
    ymin = np.nanmin(y1) * fac1
    ymax = np.nanmax(y1) * fac1
    yr = ymax - ymin
    pl.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)
    ax2 = doaxes(ee, 1, ny, 0, 1, sharex = ax1)
    for i in np.arange(M):
        pl.plot(tt, y2[i,:] * fac2, m1[i], c = col[i])
    pl.ylabel(r"$\Delta V$ (m/s)")
    ymin = np.nanmin(y2) * fac2
    ymax = np.nanmax(y2) * fac2
    yr = ymax - ymin
    pl.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)
    if y3 != None:
        ax3 = doaxes(ee, 1, ny, 0, 2, sharex = ax1)
        for i in np.arange(M):
            pl.plot(tt, y3[i,:] * fac2, m1[i], c = col[i])
        pl.ylabel(r"$V_{\rm{bis}}$ (m/s)")
        ymin = np.nanmin(y3) * fac2
        ymax = np.nanmax(y3) * fac2
        yr = ymax - ymin
        pl.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)
    pl.xlabel(xttl)
    pl.xlim(xr[0], xr[1])
    if savefile: pl.savefig(savefile)
    return

def plotPer(time, y1, y2, y3 = None, figno = 2, \
            savefile = None, period = None, fmp = 8):
    '''Plot light curve and RV amplitude spectra'''
    M, N = np.shape(y1)
    pmax = 2* (np.nanmax(time) - np.nanmin(time))
    if period == None:
        dt = np.median(time[1:]-time[:N-1])
        pmin = dt * 2.
    else:
        pmin = period / fmp
    nper = 1000
    if period == None:
        fac = 1.0
    else:
        fac = period
    if y3 is None:
        ny = 2
    else:
        ny = 3
    y = np.zeros((M*ny, N))
    y[:M,:] = y1
    y[M:2*M,:] = y2
    if not y3 is None:
        y[2*M:,:] = y3
    res = sinefitm(time, y, fmin = 1./pmax, fmax = 1./pmin, \
                   nfreq = nper)
    freq, amps, ampc = res[0], res[2], res[3]
    pers = 1.0 / freq
    amp = np.sqrt(amps**2 + ampc**2)
    amp1 = amp[:M,:]
    amp2 = amp[M:2*M,:]
    if not y3 is None:
        amp3 = amp[2*M:,:]
    
    ee = dofig(figno, 1, ny, aspect = 1)
    ax1 = doaxes(ee, 1, ny, 0, 0)
    pl.setp(ax1.get_xticklabels(), visible = False)
    pl.ylabel(r"$A_F$ (\%)")
    for i in np.arange(M):
        pl.plot(fac / pers, amp1[i,:] * fac1, ls[i], c = col[i])
    pl.ylim(0, 1.1 * np.nanmax(amp1) * fac1)    
    ax2 = doaxes(ee, 1, ny, 0, 1, sharex = ax1)
    pl.ylabel(r"$A_V$ (m/s)")
    for i in np.arange(M):
        pl.plot(fac / pers, amp2[i,:] * fac2, ls[i], c = col[i])
    pl.ylim(0, 1.1 * np.nanmax(amp2) * fac2)    
    if y3 != None:
        ax3 = doaxes(ee, 1, ny, 0, 2, sharex = ax1)
        pl.ylabel(r"$A_{\mathrm{bis}}$ (m/s)")
        for i in np.arange(M):
            pl.plot(fac / pers, amp3[i,:] * fac3, ls[i], c = col[i])
        pl.ylim(0, 1.1 * np.nanmax(amp3) * fac3)    
    if period == None:
        pl.xlabel(r"Frequency (cycles/day)")
    else:
        pl.xlabel(r"Frequency (cycles/$P_{\mathrm{rot}}^{-1}$)")
    if savefile:
        pl.savefig(savefile)
    return
    
def plotTSPer(time, y1, y2, y3 = None, figno = [1,2], savefile = [None, None], \
              discrete = False, period = None, xper = False, \
              fmp = 8):
    '''Plot both time series and amplitude spectra for light and RV'''
    plotTS(time, y1, y2, y3 = y3, figno = figno[0], discrete = discrete, \
                  savefile = savefile[0], period = period, xper = xper)
    plotPer(time, y1, y2, y3 = y3, figno = figno[1], savefile = savefile[1], \
                period = period, fmp = fmp)
    return
