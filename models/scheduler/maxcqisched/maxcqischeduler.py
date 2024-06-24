#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''A max CQI scheduler; overwrites libsimnet.basenode.Scheduler.
'''

# import concrete classes overwritten from abstract classes
from libsimnet.basenode import Scheduler

# Python imports
import sys
import math

class Scheduler(Scheduler):
    '''A max CQI resource scheduler, assigns resources based on max CQI.
    '''

    def __init__(self, ul_dl="DL", debug=False):
        '''Constructor.

        @param debug: if True prints messages
        '''
        super().__init__()
        self.debug = debug
        '''If True prints debug messages.'''
        self.ul_dl = ul_dl
        '''Indicates whether it's uplink or downlink.'''
        return

    def assign_res(self, ls_usreqs, ls_res, time_t):
        '''Assigns resources based on max CQI.

        @param ls_usreqs: a list of UserEquipment objects.
        @param ls_res: a list of Resource objects.
        @param time_t: simulation time.
        @return: a list of [user equipment, resource, ...] with assigned resources.
        '''
        self.ls_usreqs = ls_usreqs
        '''A list of UserEquipment objects.'''

        # Debug: Imprimir el estado inicial de los UserEquipments
        if self.debug:
            print(f"Time {time_t}: Starting resource assignment")
            for ue in ls_usreqs:
                chan_state = ue.chan_state
                if isinstance(chan_state, float):
                    cqi = chan_state

                print(f"UE {ue}: CQI = {cqi}")

        ls_res_tmp = list(ls_res)  # list of received resources, to be modified
        ls_usr_res = []            # user equipments with resources

        for usreq in ls_usreqs:    # for each user equipment
            ls_usr_res += [[usreq]] # initialize with UserEquipment object

        while len(ls_res_tmp) > 0:  # while there are resources available
            best_ue = None
            max_cqi = -100

            # Find the UE with the highest CQI
            for ue in ls_usreqs:
                chan_state = ue.chan_state
                if isinstance(chan_state, float):
                    cqi = chan_state

                if cqi is not None and cqi > max_cqi:
                    max_cqi = cqi
                    best_ue = ue

            if best_ue is None:
                if self.debug:
                    print("No UE with valid CQI found, exiting loop.")
                break   # no user equipments to assign resources

            n_res = self.resources_needed(best_ue, ls_res_tmp, time_t)
            n_assigned = 0  # number of resources to assign to this user eq
            res_assigned = []  # list of resources assigned to this user eq

            if n_res > 0:
                usr_res = next(ur for ur in ls_usr_res if ur[0] == best_ue)

                for res in ls_res_tmp:  # for each resources in the resource list
                    if res.res_type == best_ue.usr_grp.dc_profile["res_type"]:
                        # resource type compatible resource that can be used 
                        # in the user group
                        res_assigned.append(res)
                        n_assigned += 1
                        if n_assigned >= \
                                min(best_ue.usr_grp.dc_profile["lim_res"], n_res):
                            break

                for res in res_assigned:
                    # remove from list of resources those assigned to this user eq
                    ls_res_tmp.remove(res)
                usr_res += res_assigned

            if len(res_assigned) == 0:
                if self.debug:
                    print("No resources assigned, exiting loop.")
                break

            if len(ls_res_tmp) <= 0:  # all resources have been assigned
                break

        # Imprimir un mensaje en la salida estándar (consola) sin cerrar otros archivos
        #original_stdout = sys.stdout
        #sys.stdout = sys.__stdout__
        #print("Resources assigned based on Max CQI:")
        #for ur in ls_usr_res:
        #    print(f"UE: {ur[0]}, Resources: {len(ur) - 1}, CQI: {ur[0].chan_state}, Resources available: {len(ls_res_tmp)}, packet queue: {len(ur[0].pktque_dl.ls_recvd)}")
        #sys.stdout = original_stdout

        if self.debug:
            print(f"Time {time_t}: Resource assignment complete")
        
        return ls_usr_res

    def resources_needed(self, ue, ls_res, time_t):
        '''Resources needed by an user equipment to transmit its packets.

        Evaluates the number of resources the this user equipment can use and  the number of resources the user equipment needs to transmit all the packets in its queue.
        @param ue: UserEquipment object.
        @param ls_res: list of Resource objects.
        @param time_t: simulation time.
        '''
        chan_state = ue.chan_state
        if isinstance(chan_state, float):
            cqi = chan_state

        n_res = 0
        # calculate number of resources that can be used by the user equipment
        for res in ls_res:
            if res.res_type == ue.usr_grp.dc_profile["res_type"]:
                n_res += 1
                res_temp = res
        if n_res == 0:
            return 0
        
        nr_syms = res_temp.get_symbols()  # get number of symbols
        tb_size = ue.tr_blk.get_tb_size(nr_syms, cqi, nr_res=1, ul_dl=self.ul_dl)
        pktque_dl = ue.pktque_dl
        bits_inque = 0

        for packet in pktque_dl.ls_recvd:
            bits_inque += pktque_dl.size_bits(packet)
        # print bits inque
        #original_stdout = sys.stdout
        #sys.stdout = sys.__stdout__
        #print("bits_inque:")
        #print(f"UE: {ue}, bits_inque: {bits_inque}, Resources available: {n_res}, CQI: {cqi}, Packet queue: {len(pktque_dl.ls_recvd)}")
        #sys.stdout = original_stdout

        if bits_inque == 0:
            return 0

        res_need = math.ceil(bits_inque / tb_size)
        if res_need < n_res:
            ret = res_need + 1
        else:
            ret = n_res

        #original_stdout = sys.stdout
        #sys.stdout = sys.__stdout__
        #print("Info based on Max CQI:")
        #print(f"UE: {ue}, Resources needed: {ret}, Resources available: {n_res}, CQI: {cqi}, Packet queue: {len(pktque_dl.ls_recvd)}")
        #sys.stdout = original_stdout
        return ret

    def __str__(self):
        '''For pretty printing, on overwritten class.'''
        msg = super().__str__()
        msg += ", overwritten for max CQI example"
        return msg
