from .jitbs import *

from scipy.integrate import quad
import numpy as np

#### Frechet bs2bs
# first price
@jit
def first_price_numba_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale):
    """change h_t for frechet"""
    h_t = - kappa * np.log(1 - np.exp(-(t/scale) ** (-alpha)))
    k_star = k * np.exp(h_t-(rd-rf)*t)
    term1 = np.exp(- rf * t) * np.exp(-h_t) * black_scholes_numba(s, k_star, t, 0, sigma0)
    term2 = 1 - np.exp(-(t/scale)**(-alpha)) # 1-F(t)
    return term1*term2

# second price NO JIT
def second_price_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale):
    def integrand(tau_s):
        h_t = - kappa * np.log(1 - np.exp(-(tau_s/scale) ** (-alpha)))
        weight = (alpha/scale) * (tau_s/scale)**(-1-alpha) * np.exp(-(tau_s/scale)**(-alpha))
        bsprice = black_scholes_numba(s*np.exp(-h_t)*(1+kappa),
                                k*np.exp(-(rd-rf)*t),
                                1,
                                0, 
                                (sigma0 ** 2 * tau_s + sigma1 ** 2 * (t-tau_s)) ** 0.5)
        return weight * bsprice
    factor2, err = quad(integrand, 0, t)
    return np.exp(-rf*t)*factor2

def price_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale):
    """final price"""
    price1 = first_price_numba_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale)
    price2 = second_price_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale)
    return price1 + price2

#### Approx Frechet bs2bs
@jit
def second_price_f_aprx(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale):
    term1 = black_scholes_numba(s*(1+kappa), k*np.exp(-(rd-rf)*t), t, 0, sigma1)
    term2 = np.exp(-(t/scale)**(-alpha)) # F(t)
    return term1*term2 

def price_f_aprx(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale):
    p1 = first_price_numba_f(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale)
    p2 = second_price_f_aprx(s, k, t, rd, rf, sigma0, sigma1, alpha, kappa, scale)
    return p1 + p2

#### Poisson bs2bs
# first price
@jit
def first_price_numba_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa):
    """change h_t for poisson"""
    h_t = kappa * lamb * t # kap*h(t)
    k_star = k * np.exp(h_t-(rd-rf)*t)
    term1 = np.exp(- rf * t) * np.exp(-h_t) * black_scholes_numba(s, k_star, t, 0, sigma0)
    term2 = np.exp(-lamb*t) # 1-F(t)!!!
    return term1*term2

def second_price_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa):
    def integrand(tau_s):
        h_t = kappa * lamb * tau_s
        weight = lamb*np.exp(-lamb*tau_s)
        bsprice = black_scholes_numba(s*np.exp(-h_t)*(1+kappa),
                                k*np.exp(-(rd-rf)*t),
                                1,
                                0, 
                                (sigma0 ** 2 * tau_s + sigma1 ** 2 * (t-tau_s)) ** 0.5)
        return weight * bsprice
    factor2, err = quad(integrand, 0, t)
    return np.exp(-rf*t)*factor2

def price_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa):
    """final price"""
    price1 = first_price_numba_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa)
    price2 = second_price_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa)
    return price1 + price2


#### Approx Poisson bs2bs
@jit
def second_price_po_aprx(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa):
    term1 = black_scholes_numba(s*(1+kappa), k*np.exp(-(rd-rf)*t), t, 0, sigma1)
    term2 = 1.-np.exp(-lamb*t)
    return term1*term2
@jit
def price_po_aprx(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa):
    """final price"""
    price1 = first_price_numba_po(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa)
    price2 = second_price_po_aprx(s, k, t, rd, rf, sigma0, sigma1, lamb, kappa)
    return price1 + price2


























