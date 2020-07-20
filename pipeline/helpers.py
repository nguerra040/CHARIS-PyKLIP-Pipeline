import os
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy

# adapted from Currie's IDL code
def get_star_spot_ratio(mod, wvs, mjd, manual=None):
    central_wv = 1.550
    mod = int(mod)
    wvs = np.array(wvs)

    if manual == None:
        if mjd < 57995:
            if mod == 50:
                atten_factor = 10**(-0.4 * 5.991)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.06 / 1.0857 * atten_factor
            elif mod == 25:
                atten_factor = 10**(-0.4 * 7.5037)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.06 / 1.0857 * atten_factor
            else:
                atten_factor = (mod / 25)**2 * 10**(-0.4 * 7.5037)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.06 / 1.0857 * atten_factor
        else:
            if mod == 50:
                atten_factor = 10**(-0.4 * 4.92)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.05 / 1.0857 * atten_factor
            elif mod == 25:
                atten_factor = 10**(-0.4 * 6.412)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.05 / 1.0857 * atten_factor
            else:
                atten_factor = (mod / 25)**2 * 10**(-0.4 * 6.412)
                conv_factor = (central_wv / wvs)**2
                atten_factor = atten_factor * conv_factor
                error = 0.05 / 1.0857 * atten_factor
    else:
        atten_factor = manual
        conv_factor = (central_wv / wvs) ** 2
        atten_factor = atten_factor * conv_factor
        error = 0.05 / 1.0857 * atten_factor

    return unumpy.uarray(atten_factor, error)



def boolean(string):
    if string == 'True' or string == 'true':
        return True
    elif string == 'False' or string == 'false':
        return False
    elif string == 'None' or string == 'none':
        return None
    else:
        raise Exception('expect true or false string.')

def get_bash_path(p):
    path = p.replace(' ', '\\ ')
    return path
