#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''A proportional fair scheduler; overwrites libsimnet.basenode.Scheduler.
'''

# import concrete classes overwritten from abstract classes
from libsimnet.basenode import Scheduler

# Python imports
import sys
import math

class Scheduler(Scheduler):
    '''A proportional fair resource scheduler based on TBS.
    '''

    def __init__(self, ul_dl="DL", debug=False):
        '''Constructor.

        @param debug: if True prints messages
        '''
        super().__init__()
        self.debug = debug
        self.ul_dl = ul_dl
        self.average_throughputs = {}
        '''Dictionary to store average throughputs for each UE.'''
        return

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on the proportional fair metric.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs

        # Initialize average throughputs if not already done
        for ue in ls_usreqs:
            if ue not in self.average_throughputs:
                self.average_throughputs[ue] = 0.0

        ls_res_tmp = list(ls_res)  # list of received resources, to be modified
        ls_usr_res = []            # user equipments with resources

        for usreq in ls_usreqs:    # for each user equipment
            ls_usr_res += [[usreq]] # initialize with UserEquipment object

        while len(ls_res_tmp) > 0:  # while there are resources available
            best_ue = None
            highest_pf_metric = -1

            # Find the UE with the highest PF metric based on TBS
            for ue in ls_usreqs:
                tbs = self.calculate_tbs(ue, ls_res_tmp)
                if tbs > 0:
                    avg_throughput = self.average_throughputs.get(ue, 1e-9)  # avoid division by zero
                    pf_metric = tbs / avg_throughput if avg_throughput > 0 else tbs

                    if pf_metric > highest_pf_metric:
                        highest_pf_metric = pf_metric
                        best_ue = ue

            if best_ue is None:
                if self.debug:
                    print("No UE with valid TBS found, exiting loop.")
                break

            n_res = self.resources_needed(best_ue, ls_res_tmp, time_t)
            n_assigned = 0
            res_assigned = []

            if n_res > 0:
                usr_res = next(ur for ur in ls_usr_res if ur[0] == best_ue)

                for res in ls_res_tmp:
                    if res.res_type == best_ue.usr_grp.dc_profile["res_type"]:
                        res_assigned.append(res)
                        n_assigned += 1
                        if n_assigned >= min(best_ue.usr_grp.dc_profile["lim_res"], n_res):
                            break

                for res in res_assigned:
                    ls_res_tmp.remove(res)
                usr_res += res_assigned

                # Update average throughput
                self.update_average_throughput(best_ue, res_assigned, time_t)

            if len(res_assigned) == 0:
                if self.debug:
                    print("No resources assigned, exiting loop.")
                break

            if len(ls_res_tmp) <= 0:
                break

        # Print message to standard output
        #original_stdout = sys.stdout
        #sys.stdout = sys.__stdout__
        #print("Resources assigned based on Proportional Fair scheduling using TBS:")
        #for ur in ls_usr_res:
        #    print(f"UE: {ur[0]}, Resources: {len(ur) - 1}")
        #sys.stdout = original_stdout

        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")

        return ls_usr_res

    def calculate_tbs(self, ue, ls_res):
        '''Calculate the Transport Block Size (TBS) for a given UE.

        @param ue: UserEquipment object.
        @param ls_res: list of Resource objects.
        @return: TBS for the UE.
        '''
        chan_state = ue.chan_state
        if isinstance(chan_state, float):
            cqi = chan_state
        else:
            cqi = getattr(chan_state, 'cqi', None)

        if cqi is None:
            if self.debug:
                print(f"UE {ue} has no valid CQI")
            return 0

        for res in ls_res:
            if res.res_type == ue.usr_grp.dc_profile["res_type"]:
                nr_syms = res.get_symbols()
                tbs = ue.tr_blk.get_tb_size(nr_syms, cqi, nr_res=1, ul_dl=self.ul_dl)
                return tbs

        return 0

    def resources_needed(self, ue, ls_res, time_t):
        '''Resources needed by an user equipment to transmit its packets.

        @param ue: UserEquipment object.
        @param ls_res: list of Resource objects.
        @param time_t: simulation time.
        '''
        tbs = self.calculate_tbs(ue, ls_res)
        if tbs == 0:
            return 0

        pktque_dl = ue.pktque_dl
        bits_inque = 0

        for packet in pktque_dl.ls_recvd:
            bits_inque += pktque_dl.size_bits(packet)
        if bits_inque == 0:
            return 0

        res_need = math.ceil(bits_inque / tbs)
        if res_need < len(ls_res):
            ret = res_need + 1
        else:
            ret = len(ls_res)
        return ret

    def update_average_throughput(self, ue, res_assigned, time_t):
        '''Update the average throughput for a given UE.

        @param ue: UserEquipment object.
        @param res_assigned: list of Resource objects assigned to the UE.
        @param time_t: simulation time.
        '''
        tbs = self.calculate_tbs(ue, res_assigned)
        throughput = tbs * len(res_assigned)

        # Update moving average throughput
        alpha = 0.1  # smoothing factor
        self.average_throughputs[ue] = (1 - alpha) * self.average_throughputs.get(ue, 0) + alpha * throughput

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for proportional fair example using TBS"
        return msg