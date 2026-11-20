import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.constants import mu_0

def get_iota(vmec):
    """
    Return iota profile.

    The returned array is defined on the full vmec s grid.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.

    Returns
    -------
    iota : array_like
        iota profile
    """
    iota = vmec.wout.iotaf
    return iota


def get_min_iota(vmec, s_min, s_max):
    """
    Get the minimum of iota in the range s_min <= s <= s_max.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.
    s_min : float
        Minimum value of s to consider.
    s_max : float
        Maximum value of s to consider.

    Returns
    -------
    min_iota : float
        The smallest value of iota in the specified range.
    """
    s = vmec.s_full_grid
    iota = get_iota(vmec)
    mask = (s >= s_min) & (s <= s_max)
    return np.min(iota[mask])


def get_max_iota(vmec, s_min, s_max):
    """
    Get the maximum of iota in the range s_min <= s <= s_max.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.
    s_min : float
        Minimum value of s to consider.
    s_max : float
        Maximum value of s to consider.

    Returns
    -------
    max_iota : float
        The largest value of iota in the specified range.
    """
    s = vmec.s_full_grid
    iota = get_iota(vmec)
    mask = (s >= s_min) & (s <= s_max)
    return np.max(iota[mask])

def get_iota_monotonicity(vmec, s_min, s_max, thresh=1e-3):
    """
    Get the fraction of iota that is monotonic in the range s_min <= s <= s_max.

    Parameters
    ----------
    vmec : Vmec
        A simsopt or vmec++ Vmec object that has already been run.
    s_min : float
        Minimum value of s to consider.
    s_max : float
        Maximum value of s to consider.

    Returns
    -------
    ratio : float
        The fraction of iota that does NOT have the dominant sign of shear (minimize to make monotonic)
    """
    s = vmec.s_full_grid
    ds = 1/(len(s) - 1) # use full grid for ds. Should not matter but this is true and tried.
    iota = get_iota(vmec)
    mask = (s >= s_min) & (s <= s_max)
    iota_masked = iota[mask]
    ns = len(iota_masked)
    diota = np.diff(iota_masked)
    # values with |diota/ds| lower than thresh are always excluded,
    # so will be avoided by optimization
    pos = np.sum(diota>thresh * ds)
    neg = np.sum(diota<-thresh * ds)
    if pos >= neg:
        ratio = 1-pos/ns
    else:
        ratio = 1-neg/ns

    return ratio

