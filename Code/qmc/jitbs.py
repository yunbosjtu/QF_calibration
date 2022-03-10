import math
from scipy.optimize import root, brentq, minimize
from numba import jit

# numba BS
@jit
def cnd_numba(d):
    """cdf of Normal distribution"""
    
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val

@jit
def black_scholes_numba(stockPrice, optionStrike,
                        optionYears, Riskfree, Volatility):
    """Standard BS formuls"""

    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_numba(d1)
    cndd2 = cnd_numba(d2)

    expRT = math.exp((-1. * R) * T)
    callResult = (S * cndd1 - X * expRT * cndd2)

    return callResult

### for calibration
@jit
def garman_kohlhagen_numba(s, k, t, rd, rf, sigma):
    sqrt = math.sqrt(t)
    d1 = (math.log(s / k) + (rd-rf + 0.5 * sigma * sigma) * t) / (sigma * sqrt)
    d2 = d1 - sigma * sqrt
    cndd1 = cnd_numba(d1)
    cndd2 = cnd_numba(d2)

    exprdt = math.exp((-1. * rd) * t)
    exprft = math.exp((-1. * rf) * t)
    callresult = (s *exprft* cndd1 - k * exprdt * cndd2)

    return callresult

def impliedvol_func(s, k, t, rd, rf, price):
    def root_func(x):
        return garman_kohlhagen_numba(s, k, t, rd, rf, x) - price
    result = 0
    try:
        result = brentq(root_func, 0.001, 30)
    except Exception:
        print('There is a problem here with brentq')
        result = root(root_func, 0.2).x[0]
    return result