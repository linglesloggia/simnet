#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# rr_scheduler: round robin scheduler
#

'''A round robin scheduler; overwrites libsimnet.basenode.Scheduler.
'''


# import concrete classes overwritten from abstract classes
from libsimnet.basenode import Scheduler

# Python imports
import math


class Scheduler(Scheduler):
    '''A round robin resource scheduler, assigns resources in round robin.
    '''


    def __init__(self, ul_dl ="DL",debug=False):
        '''Constructor.

        @param debug: if True prints messages
        '''
        super().__init__()
        self.ue_pointer = 0
        '''A pointer to the next user equipment.'''
        self.ls_usreqs = []
        '''A list of user equipments.'''
        self.debug = debug
        '''If True prints debug messages.'''
        self.ul_dl = ul_dl
        return


    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Builds a list of user equipments and resources for each user.

        Receives a list of user equipment and a list of resources, goes through the list of user equipments groups and within it through each user equipment, assigns resources to each user equipment in round robin.
        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ... ] with as many resources as have been assigned to this user equipment, eventually none.
        '''
        self.ls_usreqs = ls_usreqs
        '''A list of UserEquipment objects.'''

        ls_res_tmp = list(ls_res) # list of received resources, to be modified
        ls_usr_res = []           # user equipments with resources
        #for usrgrp in ls_usrgrps:  # for each user group
        for usreq in ls_usreqs:     # for each user equipment
            ls_usr_res += [ [usreq] ]# initialize with UserEquipment object
        pointer_ini = self.ue_pointer 
        pt_last_assigned =  self.ue_pointer -1        
        while len(ls_res_tmp)>0:  # while there are resources available
           # v = [] # stores the resource assigned to one user
            if ls_usreqs:
                ue = self.ls_usreqs[self.ue_pointer]  # first user equipment
            else:
                break   # no user equipments to assign resources
            n_res = self.resources_needed(ue, ls_res_tmp, time_t)
            n_assigned = 0  # number of resources to assign to this user eq
            res_assigned = []  # list of resources assigned to this user eq
            if n_res > 0: 
                usr_res = ls_usr_res[self.ue_pointer]
                #v +=  [ue] # append the user equipment to the auxiliary list
                for res in ls_res_tmp:  # for each resources in the resource list
                    if res.res_type == ue.usr_grp.dc_profile["res_type"]: 
                        # resource type compatible resource that can be used 
                        # in the user group
                        res_assigned.append(res)  # adds resource
                        n_assigned +=1  # number of resources assigned to user eq
                        if n_assigned >= \
                                min(ue.usr_grp.dc_profile["lim_res"], n_res):
                            # resources assigned equal to maximum number that can be
                            # assiged to the user group and the number of resources
                            # needed to transmit its queue
                                break
                if len(res_assigned) > 0:
                    pt_last_assigned += 1
                    if pt_last_assigned >= len(self.ls_usreqs) :
                        pt_last_assigned = 0
                # if self.debug:
                #     print("Pointers: {}, {}, {}".\
                #         format(pointer_ini, pt_last_assigned, self.ue_pointer) )
    
                for res in res_assigned:
                    # remove from list of resources those assigned to this user eq
                    ls_res_tmp.remove(res) 
                usr_res += res_assigned   # assign the resources to the auxiliar variable
                #ls_usr_res += [ v ] # adds [the ue and] resources to the return list
            self.update_pointer()
            if self.ue_pointer == pointer_ini :
                if n_res > 0:
                    if self.debug:
                        msg = "Resources but their type cannot be used by "
                        msg += "the user equipments"
                        print(msg)
                self.ue_pointer = pt_last_assigned 
                self.update_pointer()
                break  # update the pointe to the next ue
            if len(ls_res_tmp)<=0:  # all resources have been assigned
                break     
        # if self.debug:
        #     print("----------------------------------")
        #     print("simulation time ",time_t)
        #     for ur in ls_usr_res:
        #         for obj in ur:
        #             print(obj.id_object)
        #     print("----------------------------------")
                
        return ls_usr_res
    
    def update_pointer(self):
        '''Updates pointer to user equipment.
        '''
        self.ue_pointer += 1
        if self.ue_pointer >= len(self.ls_usreqs):
            self.ue_pointer = 0


    def resources_needed(self, ue, ls_res, time_t):
        '''Resources needed by an user equipment to transmit its packets.

        Evaluates the number of resources the this user equipment can use and  the number of resources the user equipment needs to transmit all the packets in its queue.
        @param ue: UserEquipment object.
        @param ls_res: list of Resource objects.
        @param time_t: simulation time.
        '''

        chan_state = ue.chan_state
        n_res = 0
        # calculate number of resources that can be used by the user equipment
        for res in ls_res:
            if res.res_type == ue.usr_grp.dc_profile["res_type"]:
                # type of resource is compatible with the user group type
                n_res += 1 
                res_temp = res
        if n_res == 0:
            return 0
        
        nr_syms = res_temp.get_symbols()              # get number of symbols
        tb_size = ue.tr_blk.get_tb_size(nr_syms, chan_state,nr_res = 1,ul_dl = self.ul_dl)  
        # size in bits of one the TB with one resource
        pktque_dl = ue.pktque_dl
        bits_inque = 0
        # calculate the number of bits in the packet queue
        for packet in pktque_dl.ls_recvd:
            bits_inque += pktque_dl.size_bits(packet) 
        if bits_inque == 0:
            return 0
        # Using the bits of the TB of one resource and the bits in the queue
        # calculate the number of resources needed to send all the queue
        res_need = math.ceil(bits_inque/tb_size)
        if res_need < n_res:
            ret = res_need +1 # add because  math.ceil(bits_inque/tb_size) is an approx and sometimes it is a bit lower than needed
        else:
            ret = n_res
        return ret


    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for round robin example"
        return msg


