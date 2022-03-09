import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root, brentq, minimize
###
import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time.time())

def toc(fmt="Elapsed: %s s"):
    print(fmt % (time.time() - _tstart_stack.pop()))
###

def N(S):
    return norm.cdf(S)

## Define the functions outside the scope


def dplus(S, K, sigmaT):
    return np.log(S/K) / sigmaT + sigmaT * 0.5 

def dminus(S, K, sigmaT):
    return np.log(S/K) / sigmaT - sigmaT * 0.5


# Price functional for a vanilla option

def Price_func(S, K, T, Rd, Rf, sigmaT, Type = 1):
    result = 0
    if sigmaT == 0:
        result = np.maximum( Type * (S * np.exp(-Rf * T) - K * np.exp(-Rd * T)),0)
    else:
        term1 = np.exp(-Rf * T) * S * N(Type * dplus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT))
        term2 = np.exp(-Rd * T) * K * N(Type * dminus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT))
        result = Type * (term1 - term2)
    return result
Price = np.vectorize(Price_func)

# implied volatility

def Impliedvol_func(S, K, T, Rd, Rf, price, Type = 1):
    # Choose which root to take into account whether you are deep in the money or not.
    def Root(x):
        return Price(S, K, T, Rd, Rf, x * T ** 0.5, Type) - price
    result = 0
    try:
        result = brentq(Root, 0.00001, 30)
    except Exception:
        print('There is a problem here with brentq')
        result = root(Root, 0.2).x[0]
    return result
Impliedvol = np.vectorize(Impliedvol_func)

# The different delta S:pips, S:pa, F:pips, F:pa

def DeltaSpips_func(S, K, T, Rd, Rf, sigmaT, Type =1):
    return Type * np.exp(- Rf * T) * N(Type * dplus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT))

def DeltaSpa_func(S, K, T, Rd, Rf, sigmaT, Type =1):
    return Type * np.exp(- Rd * T) * N(Type * dminus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT)) * K / S

def DeltaFpips_func(S, K, T, Rd, Rf, sigmaT, Type =1):
    return Type * N(Type * dplus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT))

def DeltaFpa_func(S, K, T, Rd, Rf, sigmaT, Type =1):
    F0T = np.exp((Rd - Rf) * T) * S 
    return Type * N(Type * dminus(np.exp(-Rf * T) * S, np.exp(-Rd * T) * K, sigmaT)) * K / F0T

DeltaSpips = np.vectorize(DeltaSpips_func)
DeltaSpa = np.vectorize(DeltaSpa_func)
DeltaFpips = np.vectorize(DeltaFpips_func)
DeltaFpa = np.vectorize(DeltaFpa_func)


# Retriving KATM (only a difference between pips and pa)

def KATMpips_func(S, T, Rd, Rf, sigma):
        return S * np.exp( (Rd - Rf)*T) * np.exp( sigma ** 2 * 0.5 * T)

def KATMpa_func(S, T, Rd, Rf, sigma):
    return S * np.exp( (Rd - Rf) * T) * np.exp( - 0.5 * T * sigma ** 2)

KATMpips = np.vectorize(KATMpips_func)
KATMpa = np.vectorize(KATMpa_func)


# Retriving the Kx% S:pips, S:pa, F:pips, F:pa

def KDSpips_func(S, T, Rd, Rf, sigma, percent, Type =1):
    F0T = np.exp((Rd - Rf) * T) * S
    term1 = - Type * norm.ppf( np.exp(Rf * T) * percent) * sigma * T ** 0.5 + sigma ** 2 * T * 0.5
    return F0T *  np.exp(term1)

def KDFpips_func(S, T, Rd, Rf, sigma, percent, Type =1):
    F0T = np.exp((Rd - Rf) * T) * S
    term1 = - Type * norm.ppf( percent) * sigma * T ** 0.5 + sigma ** 2 * T * 0.5
    return F0T *  np.exp(term1)

def KDSpa_func(S, T, Rd, Rf, sigma, percent, Type = 1):
    result = 0
    Kmax = KDSpips(S, T, Rd, Rf, sigma, percent, Type)
    Kmin = 0.1
    
    def Root(x):
        return DeltaSpa(S, x, T, Rd, Rf, sigma * T ** 0.5, Type) - Type * percent
    
    def Minimize(x):
        return - DeltaSpa(S, x, T, Rd, Rf, sigma * T ** 0.5, 1)
    
    if Type == 1: #this is a put delta so there is only one solution
        Kmin = minimize(Minimize, Kmax/2).x[0]

    try:
        result = brentq(Root, Kmin, Kmax)
    except:
        result = np.nan
    return result

def KDFpa_func(S, T, Rd, Rf, sigma, percent, Type = 1):
    result = 0
    Kmax = KDFpips(S, T, Rd, Rf, sigma, percent, Type)
    Kmin = 0.1
    
    def Root(x):
        return DeltaFpa(S, x, T, Rd, Rf, sigma * T ** 0.5, Type) - Type * percent
    
    def Minimize(x):
        return - DeltaFpa(S, x, T, Rd, Rf, sigma * T ** 0.5, 1)
    
    if Type == 1: #this is a put delta so there is only one solution
        Kmin = minimize(Minimize, Kmax/2).x[0]
        
    try:
        result = brentq(Root, Kmin, Kmax)
    except:
        result = np.nan
    return result

KDSpips = np.vectorize(KDSpips_func)
KDSpa = np.vectorize(KDSpa_func)
KDFpips = np.vectorize(KDFpips_func)
KDFpa = np.vectorize(KDFpa_func)


# class defining the GK with the right conventions

S_PIPS = {'S_F' : 'S', 'pips_pa' : 'pips'}
S_PA = {'S_F' : 'S', 'pips_pa' : 'pa'}
F_PIPS = {'S_F' : 'F', 'pips_pa' : 'pips'}
F_PA = {'S_F' : 'F', 'pips_pa' : 'pa'}

CONVENTIONS  = [S_PIPS, S_PA, F_PIPS, F_PA]


class GK():
    # Initialize the class with the dictionary of conventions
    # {'S_F' : x, 'pips_pa': y}

    def __init__(self, **kwargs):
        if kwargs:
            # check if it is one of the format
            if kwargs in CONVENTIONS:
                # we are good to go
                self.conventions = kwargs
            else:
                print("There is an error in the convention format {'S_F' : x, 'pips_pa' : y}.\n x = S or F and y = pips or pa \n Falling back to default")
                self.conventions = S_PIPS
        else:
            # there are no conventions so fall back to S:pips
            self.conventions = S_PIPS
            print('No conventions provided.\n Falling back to default S:pips: ', self.conventions)

        # setting up the functions related to it
        # Functions specifying
        self.Price = Price
        self.Impliedvol = Impliedvol
        if self.conventions['pips_pa'] == 'pips':
            self.KATM = KATMpips
            if self.conventions['S_F'] == 'S':
                self.Delta = DeltaSpips
                self.KDelta = KDSpips
            elif self.conventions['S_F'] == 'F':
                self.Delta = DeltaFpips
                self.KDelta = KDFpips
        elif self.conventions['pips_pa'] == 'pa':
            self.KATM = KATMpa
            if self.conventions['S_F'] == 'S':
                self.Delta = DeltaSpa
                self.KDelta = KDSpa
            elif self.conventions['S_F'] == 'F':
                self.Delta = DeltaFpa
                self.KDelta = KDFpa

    def Info(self):
        print('The present conventions are:\nSpot vs Forward: ', self.conventions['S_F'], '\npips vs premium adjusted:', self.conventions['pips_pa'])

