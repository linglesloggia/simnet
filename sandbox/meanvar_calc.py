# meanvar_calc : calculate mean and variance over a finite moving sample
#
# attempts to calculate efficiently the mean and variance of a finite sample
# of size m which moves along a succession of values

from statistics import mean, pvariance, variance
import sys

ls_n = [0,0,0,0,0]  # a sample of fixed size 5


def mean_val_n(med, new_val):
    '''Attempts to calculate mean and M2 efficiently.

    Must subtract contribution of first element in sample, and add contribution of last element in sample.
    @param med: mean of actual sample.
    @param new_val: new value to add to sample.
    @return: (mean, m2) for new sample.
    '''
    global ls_n
    count = 5   # size of sample
    ls_n += [new_val]              # add new value to sample as last element
    old_val, ls_n = ls_n[0], ls_n[1:]  # strip off oldest element in sample
    print("    Sample", ls_n)
    # mean calculation, subtract oldest value contribution
    med = med - old_val / count
    #d2_old = (old_val - med_old)**2  # med_old not known!
    #m2 = m2 - d2_old
    # mean calculation, add new value contribution
    med = med + new_val / count
    # next correction useless, old element contribution not subtracted
    #d2 = (new_val - med)**2
    #m2 = m2 + d2
    return count, med  #, m2


# Calculate mean and variance incrementally, but for all values
#
# Welford's algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm 
#
# For a new value new_value, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far

def update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (count, mean, variance, sample_variance)


# Media and variance on a movin sample of size k
#
# https://math.stackexchange.com/questions/3112650/formula-to-recalculate-variance-after-removing-a-value-and-adding-another-one-gi 





# test function
def fn_test(vals, n_size):
    global ls_n
    ea = [0, 0, 0]    # initial existing aggregate [count, mean, m2]

    ls_n = []
    for i in range(0, n_size):
        ls_n += [0]     # a list of 0

    count_n, med_n = 0, 0.0
    for val in vals:
        ea = update(ea, val)
        count_n, med_n = mean_val_n(med_n, val)
    count, ea_mean, ea_var, ea_s_var = finalize(ea)

    # from aggregate calculation
    print("Aggregate calc: count={}, mean={}, var={}, sample_var={}".\
        format(count, round(ea_mean, 3), round(ea_var, 3), round(ea_s_var, 3) ) )
    # from finite sample calculation
    print("Sample {}: count={}, mean={}".\
        format(ls_n, count_n, round(med_n,3) ) )

    # from statistics library
    st_count = len(ls_n)
    st_mean = mean(ls_n)
    st_pvar = pvariance(ls_n, st_mean)
    st_var = variance(ls_n, st_mean)
    print("From statistics: count={}, mean={}, var={}, sample_var={}".\
        format(st_count, round(st_mean,3), round(st_pvar,3), round(st_var,3) ) )


if __name__ == "__main__":

    vals = [1,2,3,4,5]
    fn_test(vals, 5)
    vals = [4,5,3,3,1]
    fn_test(vals, 5)
