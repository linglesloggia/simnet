#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simnet_09: a very elementary simulator example
#

'''
Simple wireless communications simulator example, results module.

This module contains statistics and graphics for simulation results presentation.
'''

from libsimnet.pktqueue import init_dc_traf, show_dc_traf     # traffic counters



class Statistics:
    '''Collects data, presents statistics.
    '''

    def __init__(self, sim_obj):
        '''Constructor.

        @param sim_obj: a Simulation object to collect data from.
        '''

        self.sim_obj = sim_obj
        '''A Simulation object to collect data from.'''

        self.dc_traf = init_dc_traf()
        '''Dictionary of traffic counters.'''
        self.sim_duration = 0
        '''Duration of the simulation in seconds.'''
        return


    def show_sent_rec(self, ul_dl="DL"):
        '''Show packets sent and received during simulation.

        @param ul_dl: show packets from upload (UL) or download (DL) queue.
        '''        
        for slc in self.sim_obj.ls_slices:
            print("\nSlice {}".format(slc.id_object))
            for usreq in slc.ls_usreqs:
                # determine queue to transmit from, downlink or uplink
                if ul_dl == "DL":
                    pktque = usreq.pktque_dl
                elif ul_dl == "UL":
                    pktque = usreq.pktque_ul
                else:
                    print("Statistics.show_sent_rec: invalid traffic type", ul_dl)
                print("\n--- UserEquipment {}: packet queue {}".\
                    format(usreq.id_object, pktque.id_object) )
                print("    -- ", end=""); pktque.show_ls_pkts("Received")
                print("    -- ", end=""); pktque.show_retrans()
                print("    -- ", end=""); pktque.show_ls_pkts("Sent")
                print("    -- Counters:")
                pktque.show_counters()
        return


    def sim_totals(self, show=True):
        '''Show simulation results.

        @param show: if True, prints totals and duration.
        @return: returns dictionary of traffic counters and simulation duration. 
        '''
        #print("\n--- Simulation totals")
        for tg in self.sim_obj.ls_trfgen:
            self.dc_traf["Received"][0] += tg.pktque.dc_traf["Received"][0]
            self.dc_traf["Received"][1] += tg.pktque.dc_traf["Received"][1]
            self.dc_traf["Sent"][0] += tg.pktque.dc_traf["Sent"][0]
            self.dc_traf["Sent"][1] += tg.pktque.dc_traf["Sent"][1]
            self.dc_traf["Lost"][0] += tg.pktque.dc_traf["Lost"][0]
            self.dc_traf["Lost"][1] += tg.pktque.dc_traf["Lost"][1]
            self.dc_traf["Dropped"][0] += tg.pktque.dc_traf["Dropped"][0]
            self.dc_traf["Dropped"][1] += tg.pktque.dc_traf["Dropped"][1]
            self.dc_traf["Delay"] += tg.pktque.dc_traf["Delay"]
        if show:
            self.show_counters()
            self.sim_duration = self.sim_obj.end_time - \
                self.sim_obj.start_time
            print("Duration: simulation time {}, real time {:0.3f} s".\
                format(self.sim_obj.time_sim, self.sim_duration))
        return self.dc_traf, self.sim_duration


    def show_counters(self):
        '''Shows list indexes and counters in number of packets and bits.
        '''
        dc_keys = ["Received", "Sent", "Lost", "Dropped"] #, "Transport Blocks"]
        show_dc_traf(self.dc_traf, meanvar=False, dc_keys=dc_keys)
        return


    def show_usereq_stats(self):
        '''Show simulation results for each user equipment.
        '''

        print("Statistics per user equipment queue:")
        print("    Queue Id          bits to send     bits sent    throughput")
        #for user in self.sim_obj.ls_all_usreqs:
        for tg in self.sim_obj.ls_trfgen:
            bits_rec, bits_sent, bits_to_send = tg.pktque.get_bits()
            bps = int((bits_sent)/self.sim_duration)
            print("    {:15s}   {:12d}   {:11d}  {:12d}".\
                format(tg.pktque.id_object, bits_to_send, bits_sent, bps) )
        return



