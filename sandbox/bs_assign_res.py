#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# PyWiSim: test code for resource assignment to slices

'''Inter slice scheduler, a test code.

Assigns resources to slices according to a matrix of rows::

    [slice, resource_type, quantity]

Assigns to the indicated slice the indicated quantity of resources of the indicated type. When the desired assingment exceeds the available quantity of resources of the indicated type, nothing is done.   

This is a self contained test, with a simplified version of the required entities, just to test the assignment algorithm
'''

class Resource:
    counter = 0
    def __init__(self, res_type):
        self.res_type = res_type    # resource type, a string
        Resource.counter += 1
        self.id_res = "RS-"+self.res_type + "-" + str(Resource.counter)
        return
    def __str__(self):
        return self.id_res
    
class Slice:
    counter = 0
    def __init__(self):
        Slice.counter += 1
        self.id_object = "SL-" + str(Slice.counter)
        #self.ls_res_slc = []
        return
    def __str__(self):
        return self.id_object
    


class BaseStation:
    def __init__(self):
        self.dc_res = {}    # dict {res_type: [ [res, id_slc], ...] }
        '''Dictionary {res_type: [ [res, id_slc|None], ...] }; for each resource type provides a list of resources and to which slice each one has been assigned, or None if it has not been assigned yet.
        '''
        self.dc_slc = {}   # dict {id_slc: slice_object}
        '''Dictionary of Slice objects by slice identificator, {id_slice : Slice object}.'''
        return

    def mk_dc_res(self, res_type, res_qty):
        '''Makes Resource objects and adds to resource dictionary by type.

        @param res_type: the type of resource.
        @param res_qty: the quantity of resources of this type to create.
        '''
        ls_res_type = []
        for i in range(0, res_qty):
            new_res = Resource(res_type)
            ls_res_type += [[new_res, None]]
        self.dc_res[res_type] = ls_res_type
        return

    def show_dc_res(self):
        '''Shows dictionary of resource objects by type.
        '''
        for key in self.dc_res.keys():
            print("    {}: ".format(key), end="")
            for res, ass in self.dc_res[key]:
                print("{}:{}, ".format(res, ass), end="")
            print()
        return



class InterSliceSched:
    def __init__(self):
        self.dc_slc_res = {} # dict of of id_{slice : [resource, ...] }
        '''Dictionary {id_slice : [resource, ...] }, list of resources assigned to a slice.
        '''

    def assign_slc_res(self, dc_slc, dc_res, assgn_mat): #, dc_sl_res):
        '''Assigns resources to slices according to an assignment matrix.

        @param dc_slc: dictionary of Slice objects by slice id.
        @param dc_res: dictionary of list of Resource objects by type.
        @param assgn_mat: the resource to slice assingment matrix.
        '''
        #ls_slc_res = []   # list of [slice, [resource, ...]]
        for id_slc, res_type, qty in assgn_mat:
            q = 0
            ls_res_ass = dc_res[res_type]
            for i in range(0, len(ls_res_ass)):
                if not ls_res_ass[i][1]:
                    ls_res_ass[i][1] = id_slc
                    if id_slc in self.dc_slc_res:
                       self.dc_slc_res[id_slc] += [ls_res_ass[i][0]]
                    else:
                        self.dc_slc_res[id_slc] = [ls_res_ass[i][0]]
                    q += 1
                else:
                    pass
                if q >= qty:
                    break
        return

    def unassign_res(self, dc_res): #, dc_slc_res):
        '''Liberates resources from slices, for new assignment.
        '''
        for res_type in dc_res:
            for i in range(0, len(dc_res[res_type])):
                dc_res[res_type][i][1] = None
        self.dc_slc_res  = {}
        return
        
    def show_dc_slc_res(self):
        if not self.dc_slc_res:    # empty dictionary
            print("    No elements")       
            return
        for key, value in self.dc_slc_res.items():
            print("    {}: ".format(key), end="")
            for res in value:
                print("{}, ".format(res), end="")
            print()
        return  


if __name__ == "__main__":

    print("=== Inter slice scheduler, a test code\n")
    bs = BaseStation()                  # create BaseStation
    in_sl_sched = InterSliceSched()     # create InterSlicescheduler
    for i in range(0,3):                # create 3 slices
        sl = Slice()
        bs.dc_slc[sl.id_object] = [sl, [] ]
    print("--- Slices:", bs.dc_slc.keys())

    print("\n--> Make resources")
    bs.mk_dc_res("Good", 3)
    bs.mk_dc_res("Fair", 2)
    bs.mk_dc_res("Poor", 4)
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
    ls_res_ass = in_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    print("--- Resources dictionary, after being assigned to slices")
    bs.show_dc_res()
    print("--- List of resources and assignment")
    in_sl_sched.show_dc_slc_res()

    print("\n--> Unassign resources")
    print("--- Resources dictionary, after unassingment")
    bs.show_dc_res()
    print("--- List of resources and assignment")
    in_sl_sched.unassign_res(bs.dc_res)
    in_sl_sched.show_dc_slc_res()
        
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
    ls_res_ass = in_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    print("--- Resources dictionary, after being assigned to slices")
    bs.show_dc_res()
    print("--- List of resources and assignment")
    in_sl_sched.show_dc_slc_res()


