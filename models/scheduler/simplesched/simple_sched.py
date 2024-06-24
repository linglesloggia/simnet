#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simple_sched: overwrites for a simple scheduler
#

'''A simple scheduler model, overwritten from libsimnet.basenode.Scheduler.
'''

# import abstract classes to overwrite
from libsimnet.basenode import Scheduler


class Scheduler(Scheduler):
    '''A simple resource scheduler, overwrites Scheduler in libsimnet.
    '''


    def __init__(self):
        '''Constructor.
        '''
        super().__init__()
        self.ix_usreq_assgn = 0
        '''Index of first user equipment to assign resources. Points to the next user equipment after the last one to receive resources, i.e. the first to be considered on next resources assignment.
        '''
        return


    def assign_res(self, ls_usreqs, ls_res, time_t=0):
        '''Builds a list of user equipments and resources for each user.

        Receives a list of user equipments and a list of resources, goes through the list of user equipments and assigns resources to each user equipment according to its user group profile, channel characteristics, and possibly other factors.

        In this simple example, one resource per user equipment is assigned in round robin until all resources have been assigned. If there are more user equipments than resources, some user equipments will not receive any resources; if there are more resources than user equipments, user equipments will receive more than one resource; user equipments in the firsts places in the list are privileged. 
        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation instant time.
        @return: a list of [user equipment, resource, ... ] with as many resources as have been assigned to this user equipment, eventually none.
        '''
        ls_usr_res = []             # user equipments with resources
        #for usrgrp in ls_usrgrps:  # for each user group
        for usreq in ls_usreqs:     # for each user equipment
            ls_usr_res += [ [usreq] ]   # initialize with UserEquipment object

        # assign resources, decrement list of resources available 
        ix_res = len(ls_res) - 1    # iterator on resources
        while ix_res >= 0:          # while there are resources available

            # iterate on user equipments, assign one resource to each
            for ix_usreq in range(self.ix_usreq_assgn, len(ls_usr_res)):
                #print("time_t {}, ix {}, ix_usreq_assgn {}".\
                #    format(time_t, ix_usreq, self.ix_usreq_assgn))
                usr_res = ls_usr_res[ix_usreq]
                usreq = usr_res[0]   # recover usereq from item [usreq, res,...]
                # get user profile from usreq.usr_grp.profile
                pass
                # get channel state from usreq.chan.get_chan_state
                pass
                pkts_trans, pkts_retrans = usreq.pktque_dl.get_state()
                # if usreq has nothing to transmit, skip assignment
                if pkts_trans == 0 and pkts_retrans == 0: # no packets to transmit
                    if ix_usreq == len(ls_usr_res) - 1:   # only one usereq in list
                        return ls_usr_res
                    else:
                        continue   # no resources assigned
                else:          # apply assignment algorithm, here round robin
                    usr_res += [ ls_res[ix_res] ]       # assign one resource
                    ix_res -= 1          # decrement counter of available resources
                    if ix_res < 0:       # all resources have been assigned
                        if ix_usreq >= len(ls_usr_res) -1 :     # last in list
                            self.ix_usreq_assgn = 0     # point to first in list
                        else:       # next to receive not last in list
                            self.ix_usreq_assgn = ix_usreq + 1 # next to receive
                        return ls_usr_res # resources exhausted, suspend assignment
            self.ix_usreq_assgn = 0  # restart from first user equipment   
        return ls_usr_res

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for simple example"
        return msg