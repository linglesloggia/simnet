#!/usr/bin/env python3
#-*- coding: utf-8 -*-




def init_dc_traf():
    '''A dictionary of traffic counters.

    Defined as a module function to be used in different places.
    @return: a dictionary of traffic values.
    '''
    dc_traf = {}                   # traffic dictionary
    init_vals = [0, 0, 0.0, 0.0] # packets, bits, mean, var
    dc_traf["Received"] = init_vals
    dc_traf["Sent"] = init_vals
    dc_traf["Lost"] = init_vals
    dc_traf["Dropped"] = init_vals
    dc_traf["Delay"] = 0.0
    return dc_traf



def show_dc_traf(dc_traf):
    '''Shows dictionary of traffic counters.

    Defined as a module function to be used in different places.
    @param dc_traf: dictionary of traffic counters to show.
    '''
    print("    Traffic        Packets        Bits          Mean      Variance")
    for nm in ["Received", "Sent", "Lost", "Dropped"]:
        nr_pkts, nr_bits, bits_mean, bits_var = dc_traf[nm]
        print("    {:10s}  {:10d}  {:10d} {:13.3f} {:13.3f}".\
            format(nm, nr_pkts, nr_bits, bits_mean, bits_var))
    if dc_traf["Sent"][0] > 0:  # tere hare packets transmitted
        delay = dc_traf["Delay"] / dc_traf["Sent"][0]
        print("    Transmission delay:    {:13.3f}".format(delay))
    else:
        print("    Transmission delay: no packets transmitted yet.")
    return


dc_traf = init_dc_traf()
show_dc_traf(dc_traf)

