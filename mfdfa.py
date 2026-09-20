"""MFDFA implementation used to generate the final scalar feature table."""
import warnings
import numpy as np
import pandas as pd

  

def findscales(n):
    # Convert to array
    scale = int(n / 10)


    scale = np.exp(np.linspace(np.log(10), np.log(int(n / 10)), scale)).astype(int)
    scale = np.unique(scale)  # keep only unique

# Sanity checks (return warning for too short scale)



    return scale



def find_q(q:list):
    # Turn to list
   
    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    return q

#fluctuations = np.zeros((len(scale),len(q)))

def getwindow(signal, n, window):

    # Manuscript: N_s non-overlapping segments from each end.
    n_segments = n // window
    forward = np.asarray(signal[:n_segments * window]).reshape(n_segments, window)
    backward = np.asarray(signal[n - n_segments * window:]).reshape(n_segments, window)
    segments = np.concatenate((forward, backward), axis=0)
    return segments



def get_trends(segments, window, order=1):
    x = np.arange(window)

    coefs = np.polyfit(x[:window], segments.T, order).T

    trends = np.array([np.polyval(coefs[j], x) for j in np.arange(len(segments))])

    return trends





def _fractal_dfa_fluctuation(segments, trends, q):

    # Detrend
    detrended = segments - trends

    # F^2(s, nu) is the mean squared residual, not a second centering.
    var = np.mean(detrended ** 2, axis=1)
    if np.any(~np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("MFDFA requires finite positive segment variances for q <= 0; segments are not silently discarded.")

    # Preserve the order of q and use the logarithmic limit only at q = 0.
    is0 = q == 0
    fluctuation = np.empty(q.shape, dtype=float)
    log_var = np.log(var)
    fluctuation[is0] = np.exp(0.5 * np.mean(log_var))
    for index in np.flatnonzero(~is0):
        # Stable equivalent of [mean(var ** (q / 2))] ** (1 / q).
        terms = 0.5 * q[index] * log_var
        maximum = np.max(terms)
        log_mean = maximum + np.log(np.mean(np.exp(terms - maximum)))
        fluctuation[index] = np.exp(log_mean / q[index])
    return fluctuation





def get_slopes(scale, fluctuations, q):
    # Extract the slopes of each `q` power obtained with MFDFA to later produce
    

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError("Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))
    # Find slopes of each q-power
    for i in range(len(q)):
        # if fluctiations is zero, log2 wil encounter zero division
        old_setting = np.seterr(divide="ignore", invalid="ignore")
        slopes[i] = np.polyfit(np.log2(scale), np.log2(fluctuations[:, i]), 1)[0]
        np.seterr(**old_setting)

    return slopes





def MFDFA(ts,q,**kwags):
    
    if isinstance(ts, (np.ndarray, pd.DataFrame)) and ts.ndim > 1:
        raise ValueError(
        "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
   
    n = len(ts)
    scale = findscales(n)

    # q = 0 is supported through its logarithmic limit.
    q = find_q(q)
    if q.ndim != 1 or len(q) < 3 or np.any(np.diff(q) <= 0):
        raise ValueError("q must contain at least three strictly increasing values.")
    if not all(np.any(q == value) for value in (0, 1, 2)):
        raise ValueError("q must include 0, 1 and 2 for the reported D0, D1 and D2.")
    i0, i1, i2 = [int(np.flatnonzero(q == value)[0]) for value in (0, 1, 2)]
    

    ts= np.cumsum(ts - np.mean(ts))
    fluctuations = np.zeros((len(scale), len(q)))

    for i, window in enumerate(scale):

    # Get window
        segments = getwindow(ts, n, window)

    # Get polynomial trends
        trends = get_trends(segments, window, order=1)

    # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments, trends, q)


    slopes = get_slopes(scale, fluctuations, q)
    tau_q = q * slopes - 1

# Compute the singularity spectrum via a numerical Legendre transform
# Approximate the derivative d(tau)/dq to get alpha (singularity strength)
    alpha = np.gradient(tau_q, q, edge_order=2)
    f_alpha = q * alpha - tau_q

    results = {
            "h(q)": slopes,
            "tau(q)": tau_q,
            "alpha": alpha,
            #"C_q": C_q,
            "f_alpha": f_alpha
        }

    # Additional multifractal measures
    results.update({
        "Delta_alpha": np.nanmax(alpha) - np.nanmin(alpha),
        "Delta_f": f_alpha[np.nanargmax(alpha)] - f_alpha[np.nanargmin(alpha)],
        "f_max": f_alpha[np.nanargmax(alpha)],
        "f_min": f_alpha[np.nanargmin(alpha)],
        "alpha_max": np.nanmax(alpha),
        "alpha_min": np.nanmin(alpha),
        "mean_alpha": np.mean(alpha),
        "alpha_0": alpha[i0],
        "Delta_alpha_right": np.nanmax(alpha) - alpha[i0],
        "Delta_alpha_left": alpha[i0] - np.nanmin(alpha),
        "Delta_s": np.nanmax(alpha) - alpha[i0] - (alpha[i0] - np.nanmin(alpha)),
        "A": (-(np.nanmax(alpha) - alpha[i0]) / np.ptp(alpha)
              if np.ptp(alpha) > 0 else np.nan)
        })

    # D_q = tau(q)/(q-1), with the derivative limit at q=1.
    # alpha is the numerical derivative; alpha(0) approximates H(0).
    results["D0"] = -tau_q[i0]
    results["D1"] = alpha[i1]
    results["D2"] = tau_q[i2] / (q[i2] - 1.0)
    return results

