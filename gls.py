import numpy as np
import scipy.linalg as sla
small = np.MachAr().eps

def sinefitm(time, data, w = None, \
             fmin = None, fmax = None, nfreq = None):
    '''    
    Least squares fit of sine curve to data. Can process multiple
    time-series simlutaneously. Returns trial frequencies, reduced chi2,
    amplitudes of sine and cosine components, and dc level.
    '''
    if fmin is None:
        fmin = 1. / (np.nanmax(time) - np.nanmin(time))
    if fmax is None:
        fmax = 0.5 / np.nanmin(time[1:] - time[:-1])
    if nfreq is None:
        nfreq = int(fmax/fmin)
    freq = np.r_[fmin:fmax:nfreq*1j]
    sha = data.shape
    if len(sha) == 1:
        data = data.reshape(1, sha[0])
    elif sha[1] < sha[0]:
        data = data.reshape(sha[1], sha[0])
    nobj, nobs = data.shape
    if w == None:
        w = np.ones(nobs)
    rchi2 = np.zeros((nobj,nfreq)) + np.nan
    dc = np.zeros((nobj,nfreq)) + np.nan
    amps = np.zeros((nobj,nfreq)) + np.nan
    ampc = np.zeros((nobj,nfreq)) + np.nan
    sumw = w.sum()
    dataw = data * w
    sumdw = dataw.sum(axis=1)
    meanw = sumdw / sumw
    dc[:,0] = meanw
    ndof = float(nobs-1)
    rchi2[:,0] = (((data.T - dc[:,0])**2).T * w).sum(axis=1) / ndof
    amps[:,0] = 0.
    ampc[:,0] = 0.
    a = np.matrix(np.empty((3,3)))
    a[2,2] = sumw
    b = np.empty(3)
    ndof -= 3
    for i in np.arange(nfreq):
        arg = 2 * np.pi * freq[i] * time
        cosarg = np.cos(arg)
        sinarg = np.sin(arg)
        a[0,0] = (sinarg**2*w).sum()
        a[0,1] = (cosarg*sinarg*w).sum()
        a[0,2] = (sinarg*w).sum()
        a[1,0] = a[0,1]
        a[1,1] = (cosarg**2*w).sum()
        a[1,2] = (cosarg*w).sum()
        a[2,0] = a[0,2]
        a[2,1] = a[1,2]
        a[abs(a)<=small] = 0.
        if sla.det(a) < small: continue
        for j in np.arange(nobj):
            b[0] = (dataw[j,:].flatten()*sinarg).sum()
            b[1] = (dataw[j,:].flatten()*cosarg).sum()
            b[2] = sumdw[j]
            c = sla.solve(a, b)
            amps[j,i] = c[0]
            ampc[j,i] = c[1]
            dc[j,i] = c[2]
        fit = amps[:,i].reshape((nobj,1)) * sinarg.reshape((1,nobs)) + \
          ampc[:,i].reshape((nobj,1))  * cosarg.reshape((1,nobs)) + \
          dc[:,i].reshape((nobj,1))
        rchi2[:,i] = ((data - fit)**2 * w).sum(axis=1) / ndof
    return freq, rchi2, amps, ampc, dc
    
