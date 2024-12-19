"""Module defining tools for FCS signal analysis"""
import numpy as np
import matplotlib.pyplot as plt
def autoco(signal, var, tau):
    """ This code defines a function called  autoco  which calculates the autocorrelation coefficient of a given input array  signal .
    The autocorrelation coefficient measures the linear relationship between the values of  signal  at different time lags.
    The function takes three arguments:  signal  (the input array),  var  (the variance of  signal ), and  tau  (the time lag).

    Args:
        signal (np.array): signal to analyze
        var (float): variance of the signal for normalization
        tau (np.array): point where the autocorrelogram is estimated

    Returns:
        (np.array) autocorrelogam at tau
    """
    if tau == 0:
        tau = 1
    if tau > np.size(signal):
        print("Pb here",tau - np.size(signal))

    return np.multiply(signal[tau:], signal[:-tau]).mean() / var

def g_classic_semilogx(signal, num_bins, tau, t_max, k_max):
    """This function implements the autocorrelograms for analyzing a signal.

    Args:
        signal (np.array): signal to analyze
        num_bins (int): The number of bins to divide the signal into for analysis
        tau (np.array): An array of time delays to be used for autocorrelation calculations
        t_max (float): The maximum time value to consider for analysis
        k_max (int): The maximum number of autocorrelogramto perform by signal

    Returns:
        (np.array) list of autocorrelogram where the samples are each row
    """
    signal = np.array(signal)
    signal = signal - signal[0]
    g_list = []
    t_now = 0
    while (signal[-1] > t_max and t_now < 50):
        signal_analysed = np.histogram(signal, bins=num_bins, range=(0, t_max))[0]
        g_signal = np.zeros(k_max)
        signal_analysed = signal_analysed - signal_analysed.mean()
        var = signal_analysed.var()
        for k in range(k_max):
            g_signal[k] = autoco(signal_analysed, var, int(tau[k]))
        g_list.append(g_signal)
        signal = signal[np.sum(signal <= t_max):]
        signal -= signal[0]
        t_now += 1
    return g_list

def g_mb_theory(tau, omega_x, omega, var):
    """The given code defines the theoretical value of an autocorelogram of an FCS signal of a Brownian Motion.
    The function takes four parameters: tau, omega_x, omega, and var.
    The var parameter is expected to be a list or tuple containing two values:
    the number of particles (particle_number) and the diffusion coefficient (diffusion_coef).

    Args:
        tau (np.array):  An array of time delays to be used for function calculations
        omega_x: _description_
        omega: _description_
        var (np.array): (N,D) i.e. (mean number of particle within the illumination volume, diffusion coef)

    Returns:
        (np.array) theoritical autocorrelation at time tau for an fcs signal observing a brownian motion
    """
    diffusion_coef = var[1]
    particle_number = var[0]
    if diffusion_coef == 0:
        g_theory = 1 / particle_number
    else:
        tau_d = omega_x ** 2 / (4 * diffusion_coef)
        tau_d_prime = omega ** 2 / (4 * diffusion_coef)
        g_theory = 1 / particle_number * np.multiply(np.reciprocal(1 + tau * 1 / tau_d),
                                                 np.reciprocal(np.power(1 + tau * 1 / tau_d_prime, 0.5)))
    return g_theory

def g_fbm_theory(tau, omega_x, omega, var):
    """The given code defines the theoretical value of an autocorelogram of an FCS signal of a fractionnal Brownian Motion.
    The function takes four parameters: tau, omega_x, omega, and var.
    The var parameter is expected to be a list or tuple containing three values:
    the number of particles (particle_number), the diffusion coefficient (diffusion_coef) and the anomalous exponent(alpha).

    Args:
        tau (np.array):  An array of time delays to be used for function calculations
        omega_x: _description_
        omega: _description_
        var (np.array): (N,D,alpha) i.e. (mean number of particle within the illumination volume, diffusion coef,anomalous exponent)

    Returns:
        (np.array) theoritical autocorrelation at time tau for an fcs signal observing a fractionnal brownian motion
    """
    particle_number = var[0]
    diffusion_coef = var[1]
    alpha = var[2]
    if diffusion_coef == 0:
        g_theory = 1 / particle_number
    else:
        tau_d = omega_x ** 2 / (4 * diffusion_coef)
        tau_d_prime = omega ** 2 / (4 * diffusion_coef)
        tau_a = np.power(tau, alpha)
        g_theory = 1 / particle_number * np.multiply(np.reciprocal(1 + tau_a * 1 / tau_d),
                                                 np.reciprocal(np.power(1 + tau_a * 1 / tau_d_prime, 0.5)))
    return g_theory

def definition_set_g(t_max, points_number):
    """This is a function that return an array(tau) that contains points_number points log distributed on [0,t_max]

    Args:
        t_max (float): max of the interval in second
        points_number (int): number of points log distributed uniformaly in the interval [0,t_max]

    Returns:
        (np.array) np.array of points log distributed uniformaly in the interval [0,t_max]
    """
    if t_max > 2 * 0.15:
        efin = 0.15
    else:
        efin = 0.15
    step = 1e-5
    s_0 = np.log(step) / np.log(10)
    s_fin = (np.log(efin)) / np.log(10)
    #The points_number points where we will estimate G
    tau = np.linspace(s_0, s_fin, points_number + 1)
    for i in range(points_number + 1):
        tau[i] = pow(10, tau[i])
    return tau

def g_convol_semilogx(signal, big_tau, t_max, points_number, stride):
    """This function calculates the autocorrelation function, g(tau), of a given signal using a semilogx scale.

    Args:
        signal (np.array): signal to analyze
        big_tau (np.array): An array of time delays to be used for autocorrelation calculations
        t_max(float): max of the interval of analysis in second
        points_number (int): number of points log distributed uniformaly in the interval [0,t_max] where the autocorrelogram is performed
        stride: size of sifnal (in second) to skip in each iteration

    Returns:
        (np.array) list of autocorrelogram where the samples are each row
    """
    signal = np.array(signal)
    signal = signal - signal[0]
    g_list = []
    t_now = 0
    while (signal[-1] > t_max and t_now < 500):
        num_bins = np.sum(signal <= t_max)
        signal_analysed = np.histogram(signal, bins=num_bins, range=(0, t_max))[0]
        g_signal = np.zeros(points_number)
        signal_analysed = signal_analysed - signal_analysed.mean()
        var = signal_analysed.var()
        k = 0
        for tau in big_tau[big_tau <= t_max / 2.0][:-1]:
            g_signal[k] = autoco(signal_analysed, var, int(tau * num_bins / t_max))
            k += 1
        for tau in big_tau[big_tau > t_max / 2.0]:
            g_signal[k] = np.NaN
            k += 1
        g_list.append(g_signal)
        signal = signal[np.sum(signal <= stride):]
        signal -= signal[0]
        t_now += 1
    return g_list
