import numpy as np
'''
Compute spot flux perturbation with linear limb-darkening, according to
formalism of Dorren (1987)
'''
def calc_bigab(alpha, beta):
    '''Calculate A & B from alpha & beta (Dorren 1987)'''
    if (beta - alpha) > (np.pi / 2.): # spot out of view
        return 0.0, 0.0
    cosalpha = np.cos(alpha)
    sinalpha = np.sin(alpha)
    cosbeta = np.cos(beta)
    sinbeta = np.sin(beta)
    tanbeta = sinbeta / cosbeta
    if (beta + alpha) <= (np.pi / 2.): # spot fully visible
        delta = 0.0
        sindelta = 0.0
        cosdelta = 1.0
        zeta = 0.0
        sinzeta = 0.0
        coszeta = 1.0
    else: # spot partly visible
        cosdelta = 1.0 / np.tan(alpha) / np.tan(beta)
        delta = np.arccos(cosdelta)
        sindelta = np.sin(delta)
        sinzeta = sindelta * sinalpha
        zeta = np.arcsin(sinzeta)
    if beta <= (np.pi / 2.): 
        T = np.arctan(sinzeta * tanbeta)
    else:
        T = np.pi - np.arctan( -sinzeta * tanbeta)
    biga = zeta + (np.pi - delta) * cosbeta * sinalpha**2 - \
        sinzeta * sinbeta * cosalpha
    bigb = (1/3.) * (np.pi - delta) * \
        ( -2 * cosalpha**3 - 3 * sinbeta**2 * cosalpha * sinalpha**2) + \
        (2/3.) * (np.pi - T) + (1/6.) * sinzeta * np.sin(2 * beta) * \
        (2 - 3 * cosalpha**2)
    return biga, bigb

def calc_littleab(ustar, uspot, fratio):
    '''Calculate a & b from u_star & u_spot & F_spot/F_star (Dorren
    1987)'''
    littlea = (1 - ustar) - (1 - uspot) * fratio
    littleb = ustar - uspot * fratio
    return littlea, littleb

def dorren_F(ustar, uspot, fratio, alpha, beta):
    '''Calculate F (fraction of stellar disk hidden by spot) from
    u_star, u_spot, F_spot/F_star, alpha & beta, following Dorren (1987)'''
    biga, bigb = calc_bigab(alpha, beta)
    littlea, littleb = calc_littleab(ustar, uspot, fratio)
    F = (littlea * biga + littleb * bigb) / np.pi / \
        (1 - ustar / 3.)
    if F < 0: F = 0
    return -F

