#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_oneusr_uldl : one user equipment test wth download and upload

'''
Test of reception and transmission with only one user equipment.
'''

# import classes from the main library
from libsimnet.usernode import UserEquipment
from libsimnet.basenode import Resource, TIME_UNIT, TIME_UNIT_S
from libsimnet.pktqueue import PacketQueue

# import abstract classes to overwrite
from models.channel.randfixchan.randfix_chan import Channel
from models.transpblock.simpletrblk.simple_trblk import TransportBlock
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator

# Python imports
from random import random
import sys

def one_user(keep_pkts):
    '''Simulation with only one user equipment to test reception and transmission.

    @return: simulation instant time.
    '''

    print("\n=== One user equipment test of reception and transmission")
    print("Tests reception, transmission and retransmission of packets.")
    print("Time unit: {}; time unit in seconds: {}".\
        format(TIME_UNIT, TIME_UNIT_S) )
    #loss_prob = 0.5     # probability of losing a packet
    #nr_pkts = 13        # number of packets to generate
    print("\n--- Set simulation scenery")
    usreq = UserEquipment(None, debug=False)
    usreq.chan = Channel()
    usreq.chan_state = 100      # optimal channel state value
    usreq.tr_blk = TransportBlock()
    id_pref_dl, id_pref_ul = "DL-", "UL-"
    usreq.pktque_dl = PacketQueue(id_pref_dl, keep_pkts=keep_pkts)
    usreq.pktque_ul = PacketQueue(id_pref_ul, keep_pkts=keep_pkts)
    trfgen_dl = TrafficGenerator(usreq.pktque_dl, gen_size=80) 
    trfgen_ul = TrafficGenerator(usreq.pktque_ul, gen_size=40) 

    print("   ", usreq)
    print("       ", usreq.chan)
    print("       ", usreq.tr_blk)
    print("        DL queue:", usreq.pktque_dl)
    print("       ", trfgen_dl)
    print("        UL queue:", usreq.pktque_ul)
    print("       ", trfgen_ul)

    print("\n--- Create resources for DL y UL") 
    ls_res = [ Resource(res_type="ResDL", res_pars=[5, 2, 1], ul_dl="DL"), 
               Resource(res_type="ResUL", res_pars=[5, 1, 1], ul_dl="UL") ]
    for res in ls_res:
        print(res)

    print("\n--- Receive and transmit packets in DL and DL")
    #time_t = 0
    for time_t in range(0, 5): 
        print("-- time {:d}".format(time_t))
        # generate packets, receive in queue
        trfgen_dl.run(time_t)
        trfgen_ul.run(time_t)
        # create transport block and transmit DL
        ls_tr_blk, tbs_total = usreq.mk_ls_trblk(ls_res, time_t+1, \
            ul_dl="DL")
        print("    DL TBs", ls_tr_blk)
        for tr_blk in ls_tr_blk:
            usreq.pktque_dl.send_tb(tr_blk[0], "Sent", time_t+1)
        # create transport block and transmit UL
        ls_tr_blk, tbs_total = usreq.mk_ls_trblk(ls_res, time_t+1, \
            ul_dl="UL")
        print("    UL TBs", ls_tr_blk)
        for tr_blk in ls_tr_blk:
            usreq.pktque_ul.send_tb(tr_blk[0], "Sent", time_t+1)

    print("--- Download traffic")
    usreq.pktque_dl.show_ls_pkts("Received")
    usreq.pktque_dl.show_ls_pkts("Sent")
    usreq.pktque_dl.show_counters()
    print("--- Upload traffic")
    usreq.pktque_ul.show_ls_pkts("Received")
    usreq.pktque_ul.show_ls_pkts("Sent")
    usreq.pktque_ul.show_counters()

    return

if __name__ == "__main__":
    #print("Usage:  python3 qa_oneusreq.py [keep_pkts]")
    #print("    where optional argument keep_pkts may be:")
    #print("        y to keep all packets (default)")
    #print("        n to discard all packets")
    #print("        an integer, to keep the indicated number of packets")
    keep_pkts = True
    one_user(keep_pkts)

