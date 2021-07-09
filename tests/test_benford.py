"""Test use of the benford module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pybenford as ben


@pytest.fixture(params=[1, 2])
def digit_nb(request):
    """Return the number of digits."""
    return request.param


@pytest.fixture(params=[10, 5])
def base(request):
    """Return the mathematical bases."""
    return request.param


def test_get_theoretical_freq_benford(base, digit_nb):
    """
    Test if theoretical proportion of the first digits considered is
    correct.
    """
    # Setup
    correct_freq_digit = [[np.array([0.30103,    0.17609126, 0.12493874,
                                    0.09691001, 0.07918125, 0.06694679,
                                    0.05799195, 0.05115252, 0.0457574]),
                          np.array([0.04139269, 0.03778856, 0.03476211,
                                    0.03218468, 0.02996322, 0.02802872,
                                    0.02632894, 0.02482358, 0.0234811,
                                    0.02227639, 0.0211893,  0.02020339,
                                    0.01930516, 0.01848341, 0.01772877,
                                    0.01703334, 0.01639042, 0.01579427,
                                    0.01523997, 0.01472326, 0.01424044,
                                    0.01378828, 0.01336396, 0.01296498,
                                    0.01258913, 0.01223446, 0.01189922,
                                    0.01158187, 0.01128101, 0.01099538,
                                    0.01072387, 0.01046543, 0.01021917,
                                    0.00998422, 0.00975984, 0.00954532,
                                    0.00934003, 0.00914338, 0.00895484,
                                    0.00877392, 0.00860017, 0.00843317,
                                    0.00827253, 0.00811789, 0.00796893,
                                    0.00782534, 0.00768683, 0.00755314,
                                    0.00742402, 0.00729924, 0.00717858,
                                    0.00706185, 0.00694886, 0.00683942,
                                    0.00673338, 0.00663058, 0.00653087,
                                    0.00643411, 0.00634018, 0.00624895,
                                    0.00616031, 0.00607415, 0.00599036,
                                    0.00590886, 0.00582954, 0.00575233,
                                    0.00567713, 0.00560388, 0.00553249,
                                    0.0054629,  0.00539503, 0.00532883,
                                    0.00526424, 0.00520119, 0.00513964,
                                    0.00507953, 0.0050208,  0.00496342,
                                    0.00490733, 0.0048525,  0.00479888,
                                    0.00474644, 0.00469512, 0.00464491,
                                    0.00459575, 0.00454763, 0.0045005,
                                    0.00445434, 0.00440912, 0.0043648])],
                          [np.array([0.43067656, 0.25192964, 0.17874692,
                                     0.13864688]),
                           np.array([0.11328275, 0.0957792, 0.08296772,
                                     0.07318271, 0.06546417, 0.05921954,
                                     0.05406321, 0.04973333, 0.04604587,
                                     0.04286768, 0.04010004, 0.0376682,
                                     0.03551452, 0.03359385, 0.03187032,
                                     0.03031503, 0.02890451, 0.02761943,
                                     0.02644378, 0.02536413])]]

    # Exercise
    current_freq_digit = ben.get_theoretical_freq_benford(digit_nb, base)

    # Verify
    assert_array_almost_equal(correct_freq_digit[base % 2][digit_nb - 1],
                              current_freq_digit, 5)

    # Cleanup - None


@pytest.fixture(params=[[12, 458, 846, 7845, 25, 65, 48, 708, 201, 35],
                        [-12, -458, -846, -7845, -25, -65, -48, -708,
                         -201, -35]])
def numbers(request):
    """Return the list of numbers."""
    return request.param


@pytest.fixture(params=[1, 2])
def digit_pos(request):
    """Return positional digit in number"""
    return request.param


def test_count_digit(digit_nb, digit_pos, numbers):
    """
    Test if distribution of digits of observed data is correct for
    positif index (digit_pos) in number.
    """
    # Setup
    correct_digit_distrib = [[np.array([1, 2, 1, 2, 0, 1, 2, 1, 0]),
                              np.array([0, 1, 0, 1, 4, 0, 0, 2, 2])],
                             [np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])]]

    
    # Exercise
    current_digit_distrib = ben.count_digit(numbers, digit_nb, digit_pos,
                                            base=10)
    
    # Verify
    assert_array_almost_equal(current_digit_distrib,
                              correct_digit_distrib[digit_nb-1][digit_pos-1])

    # Cleanup - none


@pytest.fixture(params=[-3, -2])
def digit_neg(request):
    """Return positional digit in number"""
    return request.param


def test_count_digit2(digit_nb, digit_neg, numbers):
    """
    Test if distribution of digits of observed data is correct for
    negatif index (digit_neg) in number.
    """
    # Setup
    correct_digit_distrib = [[np.array([0, 1, 0, 1, 0, 0, 1, 2, 0]),
                              np.array([1, 1, 1, 3, 1, 1, 0, 0, 2])],
                             [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                              np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])]]

    
    # Exercise
    current_digit_distrib = ben.count_digit(numbers, digit_nb, digit_neg,
                                            base=10)
    
    # Verify
    assert_array_almost_equal(current_digit_distrib,
                              correct_digit_distrib[digit_nb-1][digit_neg+3])

    # Cleanup - None


@pytest.fixture(params=[8, 5])
def basis(request):
    """Return the mathematical bases."""
    return request.param


@pytest.fixture(params=[[12, 421, 144, 3142, 24, 22, 41, 104, 401, 34],
                        [-12, -421, -144, -3142, -24, -22, -41, -104, -401,
                         -34]])
def numbers2(request):
    """Return the list of numbers."""
    return request.param


def test_count_digit3(basis, digit_pos, numbers2):
    """
    Test if distribution of digits of observed data is correct for
    mathematical basis 8 and 5.
    """
    # Setup
    correct_digit_distrib = [[np.array([3, 2, 2, 3, 0, 0, 0]),
                              np.array([2, 3, 0, 3, 0, 0, 2])],
                             [np.array([3, 2, 2, 3]),
                              np.array([2, 3, 0, 5])]]

    
    # Exercise
    current_digit_distrib = ben.count_digit(numbers2, 1, digit_pos, basis)
    
    # Verify
    assert_array_almost_equal(current_digit_distrib,
                              correct_digit_distrib[basis%2][digit_pos-1])

    # Cleanup - None


def test_count_first_digit(digit_nb, numbers):
    """
    Test if distribution of the first significant digits of observed
    data is correct.
    """
    # Setup
    correct_first_digit = [np.array([1, 2, 1, 2, 0, 1, 2, 1, 0]),
                           np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                     1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]

    # Exercise
    current_first_digit = ben.count_first_digit(numbers, digit_nb)

    # Verify
    assert_array_almost_equal(current_first_digit,
                              correct_first_digit[digit_nb-1])

    # Cleanup - None


def test_normalize_first_digit():
    """
    Test if Normalize observed distribution of the first significant
    digits is correct.
    """
    # Setup
    correct_norm_first_digit = [0.02,  0.01,  0.05,  0.13,  0.14, 0.138,
                                0.156, 0.098, 0.124, 0.092, 0.042]

    # Exercise
    numbers = np.array([10, 5, 25, 65, 70, 69, 78, 49, 62, 46, 21])
    current_norm_first_digit = ben.normalize_first_digit(numbers)

    # Verify
    assert_array_almost_equal(correct_norm_first_digit,
                              current_norm_first_digit)

    # Cleanup - None


def test_calculate_pom():
    """
    Test if physical order of magnitude is correct.
    """
    # Setup
    correct_pom = 47016.3806552262
    data_obs = np.array([0.52, 12, 12055, 548, 275, 23.215, 0.2564])

    # Exercise
    current_pom = ben.calculate_pom(data_obs)

    # Verify
    assert_almost_equal(correct_pom, current_pom, 5)

    # Cleanup - None


def test_calculate_oom():
    """
    Test if order of magnitude is correct.
    """
    # Setup
    correct_oom = 4.672249193866692
    data_obs = np.array([0.52, 12, 12055, 548, 275, 23.215, 0.2564])

    # Exercise
    current_oom = ben.calculate_oom(data_obs)

    # Verify
    assert_almost_equal(correct_oom, current_oom, 10)

    # Cleanup - None


@pytest.fixture(params=[np.array([0.30, 0.18, 0.1, 0.12, 0.08,
                                  0.07, 0.06, 0.05, 0.04]),
                        np.array([0.30, 0.18, 0.1, 0.12, 0.08,
                                  0.07, 0.06, 0.05])])
def freq_obs(request):
    """Return the array of observed proportions."""
    return request.param


def test_calculate_ssd(freq_obs):
    """
    Test if sum of squares deviation is correct.
    """
    # Setup
    correct_ssd = [-1, 12.199282041547999]
    freq_theo = np.array([0.30103,    0.17609126, 0.12493874,
                         0.09691001, 0.07918125, 0.06694679,
                         0.05799195, 0.05115252, 0.0457574])

    # Exercise
    current_ssd = ben.calculate_ssd(freq_obs, freq_theo)

    # Verify
    assert_almost_equal(correct_ssd[int(sum(freq_obs))], current_ssd, 10)

    # Cleanup - None


def test_calculate_rmssd(freq_obs):
    """
    Test if root mean sum of squares deviation is correct.
    """
    # Setup
    correct_rmssd = [-1, 1.1642490207830205]
    freq_theo = np.array([0.30103,    0.17609126, 0.12493874,
                         0.09691001, 0.07918125, 0.06694679,
                         0.05799195, 0.05115252, 0.0457574])

    # Exercise
    current_rmssd = ben.calculate_rmssd(freq_obs, freq_theo)

    # Verify
    assert_almost_equal(correct_rmssd[int(sum(freq_obs))],
                        current_rmssd, 10)

    # Cleanup - None


def test_calculate_dist_hellinger(freq_obs):
    """
    Test if Helliinger distance is correct.
    """
    # Setup
    correct_dist_hell = [-1, 0.0024700738589394314]
    freq_theo = np.array([0.30103,    0.17609126, 0.12493874,
                         0.09691001, 0.07918125, 0.06694679,
                         0.05799195, 0.05115252, 0.0457574])

    # Exercise
    current_dist_hell = ben.calculate_dist_hellinger(freq_obs, freq_theo)

    # Verify
    assert_almost_equal(correct_dist_hell[int(sum(freq_obs))],
                        current_dist_hell, 10)

    # Cleanup - None


def test_calculate_dist_k_and_l(freq_obs):
    """
    Test if Kullback & Leibler distance is correct.
    """
    # Setup
    correct_dist_kl = [-1, 0.002506787620872052]
    freq_theo = np.array([0.30103,    0.17609126, 0.12493874,
                         0.09691001, 0.07918125, 0.06694679,
                         0.05799195, 0.05115252, 0.0457574])

    # Exercise
    current_dist_kl = ben.calculate_dist_k_and_l(freq_obs, freq_theo)
    # Verify
    assert_almost_equal(correct_dist_kl[int(sum(freq_obs))],
                        current_dist_kl, 10)

    # Cleanup - None


def test_chi2_test(digit_nb):
    """
    Test if Chisquare test for Benford law is correct.
    """
    # Setup
    correct_chi2 = [855.2809432704414, 935.0265195912425]
    correct_pval = [2.4902233696110595e-179, 3.453646369136125e-141]
    data_obs = np.random.choice(range(0, 1_000_000), size=2_000)
    freq_ben = ben.get_theoretical_freq_benford(digit_nb, 10)

    # Exercise
    current_chi2, current_pval = ben.chi2_test(data_obs, freq_ben, digit_nb)

    # Verify
    assert_almost_equal(correct_chi2[digit_nb-1], current_chi2, 10)
    assert_almost_equal(correct_pval[digit_nb-1], current_pval, 10)

    # Cleanup - None


def test_g_test(digit_nb):
    """
    Test if G-test for Benford law is correct.
    """
    # Setup
    correct_chi2 = [765.7703321705173, 873.6124001871107]
    correct_pval = [4.892679461548961e-160, 3.924980609988282e-129]
    data_obs = np.random.choice(range(0, 1_000_000), size=2_000)
    freq_ben = ben.get_theoretical_freq_benford(digit_nb, 10)

    # Exercise
    current_chi2, current_pval = ben.g_test(data_obs, freq_ben, digit_nb)

    # Verify
    assert_almost_equal(correct_chi2[digit_nb-1], current_chi2, 10)
    assert_almost_equal(correct_pval[digit_nb-1], current_pval, 10)

    # Cleanup - None

@pytest.fixture(params=[np.array([0.30, 0.18, 0.1, 0.12, 0.08,
                                  0.07, 0.06, 0.05, 0.04]),
                        np.array([0.12, 0.11, 0.11, 0.11, 0.11,
                                  0.11, 0.11, 0.11, 0.11])])
def freq_obs2(request):
    """Return the array of observed proportions."""
    return request.param


def test_ks_test(freq_obs2):
    """
    Tesst if ks_test for Benford law is correct.
    """
    #Setup
    correct_ks_val = [0.2620599913279623, 0.022059991327962325]
    correct_crit_val = [0.04300697617828996, 0.04300697617828996]
    freq_ben = ben.get_theoretical_freq_benford(digit_nb=1, base=10)

    # Exercise
    current_ks_val, current_crit_val = ben.ks_test(freq_obs2, freq_ben, 1000)

    # Verify
    assert_almost_equal(correct_ks_val[int(sum(freq_obs2))],
                        current_ks_val, 10)
    assert_almost_equal(correct_crit_val[int(sum(freq_obs2))],
                        current_crit_val, 10)

    # Cleanup - None


@pytest.mark.parametrize("digit_nb", [1, 2])
@pytest.mark.parametrize("test_type", [1, 0])
def test_calculate_bootstrap_chi2(digit_nb, test_type):
    """
    Test if Average of calculated chi2 and asociate p_value is correct.
    """
    # Setup
    correct_chi2 = [[392.99027241755425, 473.478099617722],
                    [416.20277536604596, 517.6132385572166]]
    correct_pval = [[5.912866885772256e-80, 8.960125311626065e-54],
                    [6.392416145101516e-85, 1.1066703236030742e-61]]
    data_obs = np.random.choice(range(0, 1_000_000), size=2_000)
    freq_ben = ben.get_theoretical_freq_benford(digit_nb, 10)

    # Exercise
    chi2, pval = ben.calculate_bootstrap_chi2(data_obs, freq_ben, digit_nb,
                                              type_test=test_type)

    # Verify
    assert_almost_equal(correct_chi2[test_type][digit_nb-1], chi2, 10)
    assert_almost_equal(correct_pval[test_type][digit_nb-1], pval, 10)

    # Cleanup - None

def test_calculate_bootstrap_ks(digit_nb):
    """
    Test if Average of calculated ks value is correct.
    """
    # Setup
    correct_ks = [354.4254124294599, 532.5265521182702]
    correct_crit_ks = [1.02850282150919e-71, 2.1868584634807427e-64]
    data_obs = np.random.choice(range(0, 1_000_000), size=2_000)
    freq_ben = ben.get_theoretical_freq_benford(digit_nb, 10)

    # Exercise
    ks, crit_ks = ben.calculate_bootstrap_chi2(data_obs, freq_ben, digit_nb)

    # Verify
    assert_almost_equal(correct_ks[digit_nb-1], ks, 10)
    assert_almost_equal(correct_crit_ks[digit_nb-1], crit_ks)

    # Cleanup - None


if __name__ == "__main__":
    print("\nThis is test script for benford module.\n"
          "Enter : pytest\n        pytest --cov-report term-missing --cov"
          " (with coverage)\nTo test the benford module\n")
