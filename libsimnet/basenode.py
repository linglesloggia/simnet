#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# basenode.py: PyWiSim radio base station related entities
#

'''
PyWiSim, radio base station related entities module.
'''


from time import sleep, perf_counter
from abc import ABC, abstractmethod

from libsimnet.libutils import mutex_prt

# simulator constants
TIME_UNIT = "ms"
'''Time unit name.'''
TIME_UNIT_S = 0.001 
'''Time unit equivalence in seconds.'''


class BaseStation:
    '''A radio base station serving users with communications resources.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, id_base):
        '''Constructor.

        @param id_base: an identifier for the base station.
        '''

        self.id_base = id_base      
        '''An identifier for the base station.'''

        self.ls_res = []
        '''List of resources assigned to this base station.'''
        BaseStation.counter += 1    # increment counter of objects created
        self.id_object = "BS-" + str(BaseStation.counter)
        '''A unique object identifier.'''

        self.inter_sl_sched = None
        '''Inter slice scheduler associated with the base station.'''

        self.dc_res = {}    # dict {res_type: [ [res, id_slc], ...] }
        '''Dictionary {res_type: [ [res, id_slc|None], ...] }; for each resource type provides a list of resources and to which slice each one has been assigned, or None if it has not been assigned yet.'''
        self.dc_slc = {}    # dict {id_slc: slice_object}
        '''Dictionary of Slice objects by slice identificator, {id_slice : Slice object}.'''

        return


    def mk_dc_res(self, res_type, res_qty, res_pars=[1, 1, 1], res_class=None):
        '''Makes Resource objects and adds to resource dictionary by type.

        @param res_type: the type of resource.
        @param res_qty: the quantity of resources of this type to create.
        @param res_pars: Resource creation parameters.
        @param res_class: a Resource class, if it has been overwritten. If given, the Resource class received in this parameter replaces the default Resource class.
        '''
        ls_res_type = []
            #syms_slot, nr_slots, nr_sbands = res_pars
        if not res_class:
            res_class = Resource
        for i in range(0, res_qty):
            #new_res = Resource(res_type, syms_slot, nr_slots, nr_sbands)
            #new_res = Resource(res_type, res_pars)
            new_res = res_class(res_type, res_pars)
            ls_res_type += [[new_res, None]]
        if res_type in self.dc_res:
            self.dc_res[res_type] += ls_res_type
        else:
            self.dc_res[res_type] = ls_res_type
        return


    def show_dc_res(self):
        '''Shows dictionary of resource objects by type.
        '''
        for key in self.dc_res.keys():
            msg = "    {}: ".format(key)
            for res, ass in self.dc_res[key]:
                msg += "{}:{} ".format(res.id_object, ass)
            print(msg)
        return


    def show_slices(self):
        '''Show the list of slices in the base station and their resources.
        '''
        for id_slc in self.dc_slc:
            print(self.dc_slc[id_slc])
            if not self.dc_slc[id_slc].ls_res:
                print("    No resources assigned to this slice")
                continue
            for res in self.dc_slc[id_slc].ls_res:
                print("    {}".format(res))
        return


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "BaseStation {:s}, {:s}".format(self.id_base, self.id_object)
        return msg



class Slice():
    '''Represents a slice in a base station, a set of users and resources.
    '''
    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, id_slc, trans_delay=1, priority=2):
        '''Constructor.

        @param id_slc: an identifier for this slice.
        @param trans_delay: delay between transmissions.
        @param priority: order of execution.
        '''

        self.id_slc = id_slc
        '''An identifier for this object.'''
        self.trans_delay = trans_delay
        '''Delay between transmissions.'''
        self.priority = priority
        '''Order of execution.'''
        Slice.counter += 1
        self.id_object = "SL-" + str(Slice.counter)
        '''A unique object identifier.'''

        self.ls_usrgrps = []
        '''List of user groups in this slice.'''
        self.ls_usreqs = []
        '''List of user equipments in this slice.'''
        self.ls_res = []
        '''List of resources available to assign to user equipments.'''
        self.sched_dl = None
        '''Scheduler to assign download resources to user equipments,'''
        self.sched_ul = None
        '''Scheduler to assign upload resources to user equipments,'''
        self.ls_usreq_res = []
        '''List of [user equipment, resource, ... ].'''

        return


    def transmit_tbs_fin(self, usreq, time_t, res_usreq, tbs_usreq, ul_dl):
        '''Executes after slice transmit_tbs actions; method to be rewritten.

        Allows to capture state after slice transmit_tbs actions.
        @param usreq: UserEquipment object.
        @param time_t: simulation instant time.
        @param res_usreq: number of resources for this user equipment.
        @param tbs_usreq: bits included in transport blocks for this user equipment.
        @param ul_dl: transmit from upload (UL) or download (DL) queue.
        '''
        pass
        return


    def transmit_tbs(self, time_t, ul_dl="DL"):
        '''Transmits the transport blocks of each user equipment.

        For each user equipment and its assigned resources, asks user equipment to make its transport blocks and collect them in a list, determines if each transport block is lost or successfully transmitted, and informs the user equipment if the transport block was lost or not, so that the user equipment can handle retransmission.
        @param time_t: simulation instant time, for a timestamp.
        @param ul_dl: transmit from upload (UL) or download (DL) queue.
        @return: number of resources, total bits included in transport blocks, for all user equipments at time_t.
        '''
        res_total, tbs_total = 0, 0 
        ls_res = []
        for usr_res in self.ls_usreq_res:
            usreq = usr_res[0]      # the user equipment
            ls_res = usr_res[1:]    # its list of resources
            # determine queue to transmit from, downlink or uplink
            if ul_dl == "DL":
                pktque = usreq.pktque_dl
            elif ul_dl == "UL":
                pktque = usreq.pktque_ul
            else:
                print("Slice.transmit_tbs: invalid traffic type", ul_dl)
            # get TBs from user equipment
            ls_tr_blk, tbs_usreq = usreq.mk_ls_trblk(ls_res, time_t, ul_dl)
            res_total += len(ls_res)    # adds resources of this usreq
            tbs_total += tbs_usreq      # adds bits in TBs of this usreq
            # sort if transport blocks are lost in transmission
            for tr_blk in ls_tr_blk:
                sent_ok = usreq.chan.get_sent()
                # inform user equipment if transport block was lost 
                if sent_ok:
                    #usreq.pktque.send_tb(tr_blk[0], "Sent", time_t)
                    pktque.send_tb(tr_blk[0], "Sent", time_t)
                else:
                    #usreq.pktque.send_tb(tr_blk[0], "Lost", time_t)
                    pktque.send_tb(tr_blk[0], "Lost", time_t)
            self.transmit_tbs_fin(usreq, time_t, len(ls_res), tbs_usreq, ul_dl)
        return res_total, tbs_total


    def run_ini(self, time_t):
        '''Executes before slice run actions; method to be rewritten.

        Allows to capture state before slice run actions.
        @param time_t: simulation instant time.
        '''
        pass
        return


    def run_fin(self, time_t, res_total, tbs_total):
        '''Executes after slice run actions; method to be rewritten.

        Allows to capture state before slice run actions.
        @param time_t: simulation instant time.
        @param res_total: number of resources for all user equipments.
        @param tbs_total: total bits included in transport blocks for all user equipments.
        '''
        pass
        return


    def run(self, time_t):        
        '''Transmits transport blocks, returns new transmission event.

        @param time_t: simulation instant time.
        @return: next transmission event.
        '''
        self.run_ini(time_t)    # allows access to state before run actions
        ## determine channel state for each user equipment
        for usreq in self.ls_usreqs:
            usreq.pos_move(time_t)
            usreq.chan_state = usreq.chan.get_chan_state(time_t, usreq.v_pos, \
                usreq.v_vel)
        # resources and transport block counters
        res_total_dl, tbs_total_dl = 0, 0
        res_total_ul, tbs_total_ul = 0, 0
    	# assign resources in scheduler DL and transmit
        if self.sched_dl:
            ul_dl = "DL"
            self.ls_usreq_res = self.sched_dl.assign_res(self.ls_usreqs, \
                self.ls_res, time_t)
            res_total_dl, tbs_total_dl = self.transmit_tbs(time_t, ul_dl=ul_dl)
    	# assign resources in scheduler UL and transmit
        if self.sched_ul:
            ul_dl = "UL"
            self.ls_usreq_res = self.sched_ul.assign_res(self.ls_usreqs, \
                self.ls_res, time_t)
            res_total_ul, tbs_total_ul = self.transmit_tbs(time_t, ul_dl=ul_dl)
        next_event = [time_t + self.trans_delay, self.priority, self.id_object]
        res_tbs_dl = (res_total_dl, tbs_total_dl)
        res_tbs_ul = (res_total_ul, tbs_total_ul)
        self.run_fin(time_t, res_tbs_dl, res_tbs_ul)   # state after run
        ##
        return next_event


    def __str__(self):
        '''For pretty printing.
        '''

        #msg = "Slice {:s}, {:s}: \n            delay {:d}, " + \
        #    "loss prob {:.2f}, groups {:d}, res {:d}"
        msg = "Slice {:s}, {:s}:  delay {}, groups {:d}, res {:d}"
        msg = msg.format(self.id_slc, self.id_object, \
                self.trans_delay, len(self.ls_usrgrps), \
                len(self.ls_res) )
        return msg



class InterSliceSched():
    '''Inter slice scheduler, assigns resources to slices.
    '''
    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, priority=5, debug=False):
        '''Constructor.

        @param priority: order of execution.
        @param debug: if True prints messages.
        '''
        self.priority = priority
        '''Order of execution in a certain time.'''
        self.debug = debug
        '''If True shows messages.'''

        self.dc_slc_res = {}
        '''Dictionary {id_slice : [resource, ...] }, list of resources assigned to a slice.
        '''

        InterSliceSched.counter += 1
        self.id_object = "ISSched-" + str(InterSliceSched.counter)
        '''A unique object identifier.'''


    def assign_slc_res(self, dc_slc, dc_res, assgn_mat): 
        '''Assigns resources to slices according to an assignment matrix.

        @param dc_slc: dictionary of Slice objects by slice id.
        @param dc_res: dictionary of list of Resource objects by type.
        @param assgn_mat: the resource to slice assingment matrix.
        '''
        # make dictionary of resources assigned to each slice
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
        # assign list of resources to slices
        for id_slc in self.dc_slc_res:
            slc_obj = dc_slc[id_slc]
            slc_obj.ls_res = self.dc_slc_res[id_slc]
        return


    def unassign_res(self, dc_res, dc_slc): #, dc_slc_res):
        '''Releases resources from slices, for new assignment.

        @param dc_res: dictionary of list of Resource objects by type.
        @param dc_slc: dictionary of slice objects by slice id.
        '''

        # release resources from slice, set assignment to None
        for res_type in dc_res:
            ls_res = dc_res[res_type]   # list of resources of this type
            for reg in ls_res:
                reg[1] = None           # resource not assigned
        self.dc_slc_res  = {}           # nothing assigned
        # delete list of resources in slices
        for id_slc in dc_slc:
            dc_slc[id_slc].ls_res = []
        return
    

    def run(self):
        '''Runs resources assignment to slices.
        '''
        # assign resources to slices
        #   to define, now assignment is done through functions
        #   assign_res, unassign_res externally invoked.
        if self.debug:
            msg = "Inter slice scheduler {}".format(self.id_object)
            mutex_prt(msg)
        # schedule next event

        return


    def show_dc_slc_res(self):
        '''Show dictionary of resources assigned to slices.
        '''

        if not self.dc_slc_res:    # empty dictionary
            print("    No elements")       
            return
        for key, value in self.dc_slc_res.items():
            print("    {}: ".format(key), end="")
            for res in value:
                print("{} ".format(res.id_object), end="")
            print()
        return  


    def __str__(self):
        '''For pretty printing.
        '''

        msg = "InterSliceScheduler {:s}".format(self.id_object)
        return msg



class UserGroup:
    '''A group of users with the same profile.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__ (self, id_usrgr, dc_profile={}):
        '''Constructor.

        @param id_usrgr: identifier of this user group.
        @param dc_profile: a dictionary of values which define the user profile in this group.
        '''

        self.id_usrgr = id_usrgr
        '''An identifier for this user group.'''
        self. dc_profile = dc_profile
        '''Characterizes user equipments' profile in this group.'''
        UserGroup.counter += 1
        self.id_object = "UG-" + str(UserGroup.counter)
        '''A unique object identifier.'''

        self.ls_usreqs = []
        '''A list of user equipments.'''
        return


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "UserGroup {:s}, {:s}".\
                format(self.id_usrgr, self.id_object)
        if self.dc_profile:
            msg += ", profile {}".format(self.dc_profile)
        return msg



class Resource:
    '''A communications resource.

    Determines the number of symbols that can be transmitted by this resource. A Resource is a set of time slots, frequency bands, etc. May have other properties depending on the technology.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''

    
    def __init__(self, res_type="NoType", res_pars=[1, 1, 1], ul_dl="DL"):
        '''Constructor.

        @param res_type: the type of resource.
        @param res_pars: a list of parameters for resource creation.
        @param ul_dl: whether resource is for download (DL, default), upload (UL), o for download or upload (UL_DL)
        '''
        
        self.res_type = res_type
        '''The type of resource.'''
        self.syms_slot = 1
        '''Number of symbols in a slot.'''
        self.nr_slots = 1
        '''Number of slots.'''
        self.nr_sbands = 1
        '''Number of sub bands.'''
        if res_pars:
            self.syms_slot, self.nr_slots, self.nr_sbands = res_pars
        self.ul_dl = ul_dl
        '''Whether resource is for download (DL) or upload (UL).'''
        # to be further specified according to technology
        #self.freq_id = 1                # central frequency (id)
        #self.syms_slot = syms_slot      # symbols by time slot
        #self.nr_slots = nr_slots        # number of time slots
        #self.nr_sbands = nr_sbands      # number of sub-bands, (layer?)
        #
        Resource.counter += 1
        self.id_object = "RS-" + str(Resource.counter)
        '''A unique object identifier.'''

        return


    def get_symbols(self, ul_dl="DL"):
        '''Determines quantity of symbols accepted for transmission.

        @param ul_dl: whether resource is for download (DL, default), upload (UL), o for download or upload (UL_DL)
        @return: quantity of symbols accepted for transmission.
        '''

        # do required calculations, then
        nr_syms = self.syms_slot * self.nr_slots * self.nr_sbands
        return nr_syms


    def __str__(self):
        '''For pretty printing.
        '''
        symbols = self.get_symbols()
        bits = symbols * 8
        msg = "Resource {:s}, type {:s} {:s}, [{:d}, {:d}, {:d}], symbols {:4d}, bits {:5d}". \
                format(self.id_object, self.res_type, self.ul_dl, self.syms_slot, \
                    self.nr_slots, self.nr_sbands, symbols, bits) 
        return msg



class Scheduler(ABC):
    '''Assigns resources to users.
    '''
    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self):
        '''Constructor.
        '''
        Scheduler.counter += 1
        self.id_object = "SCH-" + str(Scheduler.counter)
        '''A unique object identifier.'''

        return


    @abstractmethod
    def assign_res():
        '''Builds a list of user equipments and resources for each user.

        In the concrete class, should return a list of [user equipment, resource, ... ] with as many resources as have been assigned to this user equipment, eventually none.
        '''
        pass
        return


    def show_ls_usr_res(self, ls_usr_res):
        '''Shows list of user equipments and their assigned resources.

        @param ls_usr_res: a list of [user equipment, resource, ...].
        @return: a string for printing.
        '''
        msg = ""
        for ue_res in ls_usr_res:
            msg += "    {}:".format(ue_res[0])
            for res in ue_res[1:]:
                msg += "\n        {}".format(res)
            msg += "\n"
        return msg 


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "Scheduler {:s}".format(self.id_object)
        return msg
    

