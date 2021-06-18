"""Module to verify Benford's law on observed data."""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import distributions, power_divergence

np.random.seed(2021)  # Random seed


def get_theoretical_freq_benford(nb_digit=1, base=10):
    """Theoretical proportions of Benford's law.

    Function to return the theoretical proportion of the first
    significant digits.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    nb_digit : int
        Number of first digits to consider. Default is `1`.
    base : int
        Mathematical bassis. Default is `10`.

    Returns
    ¯¯¯¯¯¯¯
    p_benford : array
        Theoretical proportion of the first digits considered.

    """
    digit = (base ** nb_digit) - (base ** (nb_digit - 1))
    p_benford = np.zeros(digit, dtype=float)
    for i in range(digit):
        p_benford[i] = (math.log((1 + (1 / (i + (base ** (nb_digit - 1))))),
                                 base))
    return p_benford


def count_first_digit(numbers, nb_digit=1):
    """Distribution of the first digits in base 10 of observed data.

    Function to return the observed distribution of the first digits
    in base 10 of an observed data set. This function removes numbers
    less than 1.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    numbers : array of numbers
        Integer array.
    nb_digit : int
        Number of first significant digits.

    Returns
    ¯¯¯¯¯¯¯
    digit_distrib : array
        Distribution of the first digits in base 10.

    """
    size_array = (10 ** nb_digit) - (10 ** (nb_digit - 1))
    # array size return
    digit_distrib = np.zeros(size_array, dtype=int)
    for number in numbers:
        number = abs(number)
        if type(number) == float:
            if number <= 9e-5:
                number = str(number)
                i = 0
                nb_string = ""
                while number[i] != 'e':
                    nb_string += number[i]
                    i += 1
                number = nb_string
            number = str(number)
            number = number.replace(".", "")
            number = number.strip("0")  # remove not-significant 0.
        if int(number) >= (10 ** (nb_digit - 1)):
            number = str(number)
            first = int(number[0:nb_digit])
            digit_distrib[first - (10 ** (nb_digit - 1))] += 1

    # nb_delet = (1 - (sum(digit_distrib)/len(numbers))) * 100
    # print(f" Warning : {nb_delet:.2f}% of numbers remove")
    return digit_distrib


def normalize_first_digit(array):
    """Normalize observed distribution of the first significant digits.

    Function normalizing an array by the sum of the array values.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    array: array of int
        Array of observed data.

    Returns
    ¯¯¯¯¯¯¯
    array: array of float
        Array of observed data normalized.

    """
    array = array / sum(array)
    return array


def build_hist_freq_ben(freq_obs, freq_theo, nb_digit, title="",
                        xlab="First digit", ylab="Proportion",
                        legend="", name_save="", size=(6, 4)):
    """Histogram of observed proportion and theoretical proportion.

    Function realizing the histogram of observed proportions and adding
    the theoretical proportion of Benford.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    freq_obs : array
        Array of observed frequency.
    freq_theo : array
        Array of theoritical frequency.
    nb_digit : int
        Number of first significant digits.
    title : string, optinal
        Title of histogram.
    xlab: string, optinal
        Label of x-axis. Default is `"First digit"`.
    ylab: string, optional
        Label of y-axis. Default is `"Proportion"`.
    legend: string, optional
        Label of the legend for the theoretical frequency.
    name_save: string, optional
        Name of the image to save in .png format,
        if you want to save it.
    size: tuple of 2 int, optional
        Plot size. Default is `(6, 4)`.

    Returns
    ¯¯¯¯¯¯¯
    Histogram.

    """
    plt.figure(figsize=size)
    plt.plot(range(1, len(freq_theo)+1), freq_theo, marker="o",
             color="red")
    plt.bar(range(1, len(freq_obs)+1), freq_obs)

    lab = []
    for i in range((10 ** (nb_digit-1)), (10 ** nb_digit)):
        lab.append(str(i))

    plt.xticks(ticks=range(1, len(freq_theo)+1), labels=lab)
    plt.title(label=title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(labels=("Benford's law", legend))
    if name_save != "":
        plt.savefig(f"{name_save}.png", transparent=True)


def calculate_pom(data_obs):
    """Physical order of magnitude.

    Function of calulated physical order of magnitude in a dataset.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    data_obs: array of int
        Interger array of observed dataset.

    Returns
    ¯¯¯¯¯¯¯
    pom : float
        Physical order of magnitude in data_obs.

    Notes
    ¯¯¯¯¯
    Benford’s Law Applications for Forensic Accounting, Auditing, and
    Fraud Detection. MARK J. NIGRINI, B.COM (HONS), MBA, PHD. 2012 by
    John Wiley & Sons, Inc. ISBN 978-1-118-15285-0

    """
    pom = max(data_obs) / min(data_obs)
    print(f"POM : {pom}")
    return pom


def calculate_oom(data_obs):
    """Order of magnitude.

    Function of calculated order of magnitude in a dataset.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    data_obs: array of int
        Interger array of observed dataset.

    Returns
    ¯¯¯¯¯¯¯
    pom : float
        Order of magnitude in data_obs.

    Notes
    ¯¯¯¯¯
    Benford’s Law Applications for Forensic Accounting, Auditing, and
    Fraud Detection. MARK J. NIGRINI, B.COM (HONS), MBA, PHD. 2012 by
    John Wiley & Sons, Inc. ISBN 978-1-118-15285-0

    """
    oom = math.log(calculate_pom(data_obs), 10)
    print(f"OOM : {oom}")
    return oom


def calculate_ssd(f_obs, f_theo):
    """Sum of squares deviation.

    Function of calculated sum of squares deviation between a observed
    proportion and a theoretical proportion.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    f_obs : array of float
        Float array of observed proportion.
        Proportion is between 0 and 1.
    f_theo : array of float
        Float array of theoretical proportion.
        Proportion is between 0 and 1.

    returns
    ¯¯¯¯¯¯¯
    sdd : float
        sum of squares deviation

    Notes
    -----
    The orginal formula uses percentage. We transforme proportion
    to percentage for the calculation.

    Benford’s Law Applications for Forensic Accounting, Auditing, and
    Fraud Detection. MARK J. NIGRINI, B.COM (HONS), MBA, PHD. 2012 by
    John Wiley & Sons, Inc. ISBN 978-1-118-15285-0

    """
    if len(f_theo) != len(f_obs):
        return -1
    sdd = sum((100*f_obs - 100*f_theo)**2)
    print(f"SDD : {sdd}")
    return sdd


def calculate_rmssd(f_obs, f_theo):
    """Root mean sum of squares deviation.

    Function of calculated root mean sum of squares deviation between
    a observed proportion and a theoretical proportion.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    f_obs : array of float
        Float array of observed proportion.
    f_theo : array of float
        Float array of theoretical proportion.

    returns
    ¯¯¯¯¯¯¯
    rmssd : float
        root mean sum of squares deviation

    """
    if len(f_theo) != len(f_obs):
        return -1
    rmssd = math.sqrt(calculate_ssd(f_obs, f_theo) / len(f_theo))
    print(f"RMSSD : {rmssd}")
    return rmssd


def calculate_dist_hellinger(f_obs, f_theo):
    """Hellinger distance.

    Function of calculated Hellinger distance between a observed
    proportion and a theoretical proportion

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    f_obs : array of float
        Float array of observed proportion.
    f_theo : array of float
        Float array of theoretical proportion.

    returns
    ¯¯¯¯¯¯¯
    dist_h : float
        Hellinger distance

    Notes
    ¯¯¯¯¯
    https://en.wikipedia.org/wiki/Hellinger_distance

    Benford’s law and geographical information – the example of
    OpenStreetMap. Mocnik, Franz-Benjamin. 2021/04/07, International
    Journal of Geographical Information Science.
    https://doi.org/10.1080/13658816.2020.1829627

    """
    if len(f_theo) != len(f_obs):
        return -1
    dist_h = math.sqrt(0.5 * (sum(np.sqrt(f_obs) - np.sqrt(f_theo)) ** 2))
    print(f"Hellinger distance : {dist_h}")
    return dist_h


def calculate_dist_k_and_l(f_obs, f_theo):
    """Kullback & Leibler distance.

    Function of calculated Kullback & Leibler distance between a
    observed proportion and a theoretical proportion

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    f_obs : array of float
        Float array of observed proportion.
    f_theo : array of float
        Float array of theoretical proportion.

    returns
    ¯¯¯¯¯¯¯
    dist_kl : float
        Kullback & Leibler distance

    Notes
    ¯¯¯¯¯
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Benford’s law and geographical information – the example of
    OpenStreetMap. Mocnik, Franz-Benjamin. 2021/04/07, International
    Journal of Geographical Information Science.
    https://doi.org/10.1080/13658816.2020.1829627

    """
    if len(f_theo) != len(f_obs):
        return -1
    dist_kl = sum(f_obs * np.log10(f_obs/f_theo))
    print(f"Kullback & Leibler distance : {dist_kl}")
    return dist_kl


def chi2_test(data_obs, f_theo, nb_digit=1):
    """Chisquare test for Benford law.

    Function performing a chisquare test of compliance to Benford law.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    data_obs : array of int
        Interger array of observed dataset.
    f_theo : array of float
        Float array of theoretical frequency.
    nb_digit : int
        Number of first siginficant digits. Default is `1`.

    Returns
    ¯¯¯¯¯¯¯
    chi2 : float
        statistics of chisquare test.
    p_avl : float
        p-value of chi2.

    """
    d_theo = np.array(f_theo * len(data_obs))
    d_obs = count_first_digit(data_obs, nb_digit)
    chi2, p_val = power_divergence(f_obs=d_obs, f_exp=d_theo, lambda_=1)
    print(f"statistics : {chi2} ; p-value : {p_val}")
    return chi2, p_val


def g_test(data_obs, f_theo, nb_digit=1):
    """G-test for Benford law.

    Function performing a G-test of compliance to Benford law.

    Parameters
    ¯¯¯¯¯¯¯¯¯
    data_obs : array of int
        Interger array of observed dataset.
    f_theo : array of float
        Float array of theoretical frequency.
    nb_digit : int
        Number of first siginficant digits. Default is `1`.

    Returns
    ¯¯¯¯¯¯¯
    g : float
        statistics of G-test.
    p_avl : float
        p-value of chi2.

    """
    d_theo = np.array(f_theo * len(data_obs))
    d_obs = count_first_digit(data_obs, nb_digit)
    print(d_obs)
    print(d_theo)
    g_stat, p_val = power_divergence(f_obs=d_obs, f_exp=d_theo, lambda_=0)
    print(f"statistics : {g_stat} ; p-value : {p_val}")
    return g_stat, p_val


def calculate_bootstrap_chi2(data_obs, f_theo, nb_digit, nb_val=1000,
                             nb_loop=1000, type_test=1):
    """Average of calculated chi2 and asociate p_value.

    Function to calculate average chi2 in the function bootstrap_chi2.

    parameters
    ¯¯¯¯¯¯¯¯¯¯
    data_obs : array of int
        Integer array of observed dataset.
    f_theo : array of float-80.72309844128006
        Float array of theoretical frequency.
    nb_digit: int
        Number of first significant digits. Default is `1`.
    nb_val : int, optinal
        Sample size. Default is `1000`.
    nb_loop : int, optional
        number of "bootstrap" procedure is performed.
        Default is `1000`.
    type_test: string or int, optional
        statistical test type performed. Default is `1`.
            String            Value   test type
            "pearson"           1     Chisquare-test.
            "log-likelihood"    0     G-test.

    Returns
    ¯¯¯¯¯¯¯
    mean_chi2: float
        Chi2 average of "bootstrap".
    p_val
        p-value of mean_chi2.
    nb_signif: int
        number of significant statistical tests in the "bootstrap"

    """
    sum_chi2 = np.zeros(nb_loop, dtype=float)
    d_theo = np.array(f_theo * nb_val)
    for i in range(nb_loop):
        ech = np.random.choice(data_obs, size=nb_val, replace=False)
        d_obs = count_first_digit(ech, nb_digit)
        result = power_divergence(f_obs=d_obs, f_exp=d_theo,
                                  lambda_=type_test)
        sum_chi2[i] = result[0]

    mean_chi2 = sum(sum_chi2) / nb_loop
    k = len(f_theo+1)
    p_val = distributions.chi2.sf(mean_chi2, k - 1)
    print(f"statistics : {mean_chi2} ; p-value : {p_val}")
    return mean_chi2, p_val


if __name__ == "__main__":
    print("\nThis is benford module. This module contains functions to"
          " analyze a data set according to Benford's law.\n")
