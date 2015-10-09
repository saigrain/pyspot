import numpy as np
from scipy.interpolate import interp1d
from dorren import dorren_F

RSUN = 6.96e8
DAY2S = 86400.0

class spots():
    """Holds parameters for spots on a given star"""
    def __init__(self, nspot):
        '''Generate fiducial parameter set for nspot spots'''
        self.nspot = nspot
        self.rstar = np.ones(nspot) * 1.0
        self.incl = np.ones(nspot) *  np.pi / 2.
        self.u = np.ones(nspot) * 0.0
        self.cspot = np.ones(nspot) * 1.0
        self.cfac = np.ones(nspot) * 0.1
        self.Q = np.ones(nspot) * 10.0
        self.vconv = np.ones(nspot) * 200.0
        self.period = np.ones(nspot) * 5.0
        self.amax = np.ones(nspot) * 0.01
        self.decay = np.zeros(nspot)
        self.pk = np.zeros(nspot)
        self.phase = np.ones(nspot) * np.pi
        self.lat = np.zeros(nspot)

    def calci(self, time, i):
        '''Calculate flux, RV and bisector span variations for one spot'''
        # Spot area
        area = np.ones(len(time)) * self.amax[i]
        if (self.pk[i] != 0) * (self.decay[i] != 0):
            tt = time - self.pk[i]
            l = tt < 0
            area[l] *= np.exp(-tt[l]**2 / 10. / self.decay[i]**2) # emergence 5 times as fast
            l = tt >= 0
            area[l] *= np.exp(-tt[l]**2 / 2. / self.decay[i]**2)  # as decay
        # Fore-shortening 
        long = 2 * np.pi * time / self.period[i] + self.phase[i]
        mu = np.cos(self.incl[i]) * np.sin(self.lat[i]) + \
            np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.cos(long)
        # Projected area 
        proj = area * mu
        proj[mu < 0] = 0
        # Flux
        if self.u[i] != 0:
            # Finite size spot with limb darkening (slow)
            N = len(time)
            spot = np.zeros(N)
            for j in np.arange(N):
                spot[j] = dorren_F(self.u[i], self.u[i], 1-self.cspot[i], \
                                       np.arcsin(np.sqrt(area[j])), \
                                       np.arccos(mu[j]))
        else:
            # Point-like spot without limb darkening
            spot = - proj * self.cspot[i]
        fac = proj * self.Q[i] * self.cfac[i] * (1 - mu)
        dF = np.copy(spot) # + fac
        # RV
        veq = 2 * np.pi * self.rstar[i] * RSUN / self.period[i] / DAY2S
        spot *= veq * np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.sin(long)
        fac = proj * self.Q[i] * mu * self.vconv[i]
        dRV = spot + fac
        bis = dRV * np.cos(long)
        return dF, dRV, bis

    def calc(self, time):
        '''Calculate flux, RV and bisector span variations for all spots'''
        N = len(time)
        M = len(self.lat)
        dF = np.zeros((M, N))
        dRV = np.zeros((M, N))
        bis = np.zeros((M, N))
        for i in np.arange(M):
            dFi, dRVi, bisi = self.calci(time, i)
            dF[i,:] = dFi
            dRV[i,:] = dRVi
            bis[i,:] = bisi
        return dF, dRV, bis

    def calci_pos(self, time, i):
        '''Calculate position for one spot'''
        # Spot area
        try:
            am = self.amax[i]
        except AttributeError:
            am = self.alphamax[i]
        if (self.pk[i] == 0) + (self.decay[i] == 0):
            area = np.ones(len(time)) * am
        else:
            area = am * \
                np.exp(-(time - self.pk[i])**2 / 2. / self.decay[i]**2)
        # Fore-shortening 
        long = 2 * np.pi * time / self.period[i] + self.phase[i]
        mu = np.cos(self.incl[i]) * np.sin(self.lat[i]) + \
            np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.cos(long)
        return area, mu, self.lat[i]

    def calc_pos(self, time):
        '''Calculate positions for all spots'''
        N = len(time)
        M = len(self.lat)
        area = np.zeros((M, N))
        mu = np.zeros((M, N))
        lat = np.zeros(M)
        for i in np.arange(M):
            dum1, dum2, dum3 = self.calci_pos(time, i)
            area[i,:] = dum1
            mu[i,:] = dum2
            lat[i] = dum3
        return area, mu, lat

def genTSreg(spots, nper = 20, npper = 1000, sum = False):
    '''Generate regularly sampled light curve, RV and bisector curves
    lasting nper periods, with npper points per period, for specified
    set of spot parameters, including evolving spots.'''
    permean = np.mean(spots.period)
    tmin = 0.0
    tmax = permean * nper 
    time = np.r_[tmin:tmax:permean/float(npper)]
    N = len(time)
    dF, dRV, bis = spots.calc(time)
    if sum == True:
        dF = np.reshape(np.sum(dF, 0), (1, N))
        dRV = np.reshape(np.sum(dRV, 0), (1, N))
        bis = np.reshape(np.sum(bis, 0), (1, N))
    return time, dF, dRV, bis

def genPosreg(spots, nper = 20, npper = 1000):
    '''Generate regularly sampled position curves
    lasting nper periods, with npper points per period, for specified
    set of spot parameters, including evolving spots.'''
    permean = np.mean(spots.period)
    tmin = 0.0
    tmax = permean * nper 
    time = np.r_[tmin:tmax:permean/float(npper)]
    N = len(time)
    area, mu, lat = spots.calc_pos(time)
    return time, area, mu, lat

def resample(time, y, tnew):
    '''Resample existing time series'''
    sha = np.shape(y)
    M = sha[0]
    N = len(time)
    y_ = np.zeros((M, N))
    for i in np.arange(M):
        g = np.interpolate.interp1d(time, y[i,:])
        y_[i,:] = g(tnew)
    return y_
