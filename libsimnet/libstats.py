#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# libstats : statistics functions library
#
'''PyWiSim statistics functions library.
'''

from statistics import mean, pvariance, variance
import sys


# Calculate mean and variance incrementally, but for all values
# Welford's algorithm, 
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm 


def meanvar_upd(existing_aggregate, new_value):
    '''Incremental mean and variance calculation, update aggregate.

    Wellford's algorithm to calculate mean and variance incrementally. This function computes the new count, new mean, new m2 for a new value, where:
        - mean accumulates the mean of the entire dataset.
        - m2 aggregates the squared distance from the mean.
        - count aggregates the number of samples seen so far.
    This function receives a new value to update an existing aggregate consisting of a tuple (count, mean, m2).
    @param existing_aggregate: a tuple (count, mean, m2).
    @param new_value: the new value to include in the incremental mean and variance calculation.
    @return: a new aggregate (count, mean, m2).
    '''
    (count, mean, m2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return (count, mean, m2)


def meanvar_fin(existing_aggregate):
    '''Incremental mean and variance calculation, finalize.

    Wellford's algorithm to calculate mean and variance incrementally. This function finalizes the calculation: it receives an existing aggregate and returns the values count, mean, variance and sample variance calculated from the received aggregate.
    @param existing_aggregate: a tuple (count, mean, m2), where count is the number of values, mean is the mean value of those count values, and m2 is (new value - mean) squared.
    @return: a tuple (count, mean, variance, sample variance).
    '''
    (count, mean, m2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, m2 / count, m2 / (count - 1))
        return (count, mean, variance, sample_variance)


# test function
def fn_test(vals, n_size):
    '''Tests Welford's algorithm against Python statistics functions.
    '''
    ea = [0, 0, 0]    # initial existing aggregate [count, mean, m2]

    ls_n = []
    for i in range(0, n_size):
        ls_n += [0]     # a list of 0

    count_n, med_n = 0, 0.0
    for val in vals:
        ea = meanvar_upd(ea, val)
    count, ea_mean, ea_var, ea_s_var = meanvar_fin(ea)

    # from aggregate calculation
    print("Aggregate calc : " +
        "count={:3d}, mean={:6.3f}, var={:6.3f}, sample_var={:6.3f}".\
        format(count, round(ea_mean, 3), round(ea_var, 3), round(ea_s_var, 3) ) )

    # from statistics library
    st_count = len(vals)
    st_mean = mean(vals)
    st_pvar = pvariance(vals, st_mean)
    st_var = variance(vals, st_mean)
    print("From statistics: " + 
        "count={:3d}, mean={:6.3f}, var={:6.3f}, sample_var={:6.3f}".\
        format(st_count, round(st_mean,3), round(st_pvar,3), round(st_var,3) ) )


if __name__ == "__main__":

    vals_1 = [1,2,3,4,5]
    fn_test(vals_1, 5)
    vals_2 = [4,5,3,3,1]
    fn_test(vals_2, 5)
    fn_test(vals_1+vals_2, 10)

