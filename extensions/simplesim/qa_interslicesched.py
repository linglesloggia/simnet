#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# PyWiSim: test code for resource assignment to slices

'''Inter slice scheduler test.

Assigns resources to slices according to a matrix of rows::

    [slice, resource_type, quantity]

Assigns to the indicated slice the indicated quantity of resources of the indicated type. When the desired assingment exceeds the available quantity of resources of the indicated type, nothing is done.   
'''

from libsimnet.basenode import Slice, Resource, BaseStation, InterSliceSched

# variables, for pydoctor
bs = None
'''BaseStation object, a radio base station.'''
in_sl_sched = None
'''InterSliceSched object, an inter slice scheduler.'''
assgn_mat = []
'''Resource assignment matrix.'''
sl = None
'''A Slice object.'''


if __name__ == "__main__":

    print("=== Inter slice scheduler, a test code\n")
    bs = BaseStation("BaseStation-1")   # create BaseStation
    in_sl_sched = InterSliceSched(4, True)     # create InterSlicescheduler
    for i in range(0,3):                # create 3 slices
        sl = Slice("Slc-", 0.0)
        bs.dc_slc[sl.id_object] = sl
    print("--- Slices:", bs.dc_slc.keys())

    print("\n--> Make resources")
    bs.mk_dc_res("Good", 3, [11, 8, 1])
    bs.mk_dc_res("Fair", 2, [11, 4, 1])
    bs.mk_dc_res("Poor", 4, [11, 2, 1])
    print("--- Resources dictionary, with available resources (not assigned)")
    bs.show_dc_res()

    print("\n--> Assign resources to slices")
    # create assignment matrix
    assgn_mat = [
        ["SL-1", "Good", 1], \
        ["SL-1", "Fair", 1], \
        ["SL-2", "Good", 1], \
        ["SL-2", "Fair", 1], \
        ["SL-3", "Good", 1], \
        ["SL-3", "Fair", 1], \
        ["SL-3", "Poor", 2], \
        ]
    print("--- Assignment matrix")
    for item in assgn_mat:
        print("    {}".format(item))
    in_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    print("--- Resources dictionary, after being assigned to slices")
    bs.show_dc_res()
    print("--- Dictionary of resources and assignment")
    in_sl_sched.show_dc_slc_res()
    print("--- Slices, list of resources assigned")
    bs.show_slices()

    print("\n--> Unassign resources")
    in_sl_sched.unassign_res(bs.dc_res, bs.dc_slc)
    print("--- Resources dictionary, after unassingment")
    bs.show_dc_res()
    print("--- Dictionary of resources and assignment")
    in_sl_sched.show_dc_slc_res()
    print("--- Slices, list of resources assigned")
    bs.show_slices()
        
    print("\n--> Assign resources to slices, new assignment")
    # create new assignment matrix
    assgn_mat = [ 
        ["SL-1", "Good", 3], \
        ["SL-2", "Fair", 2], \
        ["SL-3", "Poor", 4], \
        ]
    print("--- New assignment matrix")
    for item in assgn_mat:
        print("    {}".format(item))
    in_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    print("--- Resources dictionary, after being assigned to slices")
    bs.show_dc_res()
    print("--- Dictionary of resources and assignment")
    in_sl_sched.show_dc_slc_res()
    print("--- Slices, list of resources assigned")
    bs.show_slices()


