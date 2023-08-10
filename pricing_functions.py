import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def atm_forward(spot, term_rate, base_rate, ttm):
    """
    spot: in dom currency
    term_rate and base_rate expressed as percentage points (0.01 is 1%)
    time: in days assuming 365 days a year. This figure simplifies calculations for TTM and FX trades almost 24/7
    """
    return spot * np.exp((term_rate - base_rate)*(ttm/365))


def get_gk_price(w, forward, term_rate, base_rate, ttm, vol, strike):
    """
    Gets the price of a call option using the Garman Kohlhagen formula

    Parameters:

    w: type of option: 1 for a call and -1 for a put
    spot: Current price of the underlying asset
    term_rate: Term risk-free rate
    base_rate: Base risk-free rate
    ttm: Time to maturity
    vol: Volatility of the underlying asset
    strike: Strike price of the option
    Return value: Price of call option

    """
    T = ttm/365  # from days to years -> calculated using Time Delta

    d1 = (np.log(forward / strike) + (0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(forward / strike) - (0.5 * vol ** 2) * T) / (vol * np.sqrt(T))

    value = w*np.exp(-term_rate) * \
        (forward*norm.cdf(w*d1) - strike*norm.cdf(w*d2))
    return value


def get_strike(w, delta, forward, term_rate, base_rate, ttm, vol, minimum_tick):
    """
    Gets the strike of a call option
    w: type of option: 1 for a call and -1 for a put
    spot: in base currency
    term_rate and base_rate expressed as percentage points (0.01 is 1%)
    time: in days assuming 365 days a year
    vol: same format as rate
    """

    guess = 0.01
    threshold = 0.01
    a = 0
    b = 0
    computed_delta = 0

    while np.abs(computed_delta - delta) > threshold:
        a = get_gk_price(w, forward+minimum_tick, term_rate,
                         base_rate, ttm, vol, guess)
        b = get_gk_price(w, forward, term_rate,
                         base_rate, ttm, vol, guess)
        computed_delta = (a-b)/minimum_tick
        guess += minimum_tick
    return guess
