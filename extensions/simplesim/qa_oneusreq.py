#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_oneusreq : one user equipment test of reception and transmission

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
    loss_prob = 0.5     # probability of losing a packet
    nr_pkts = 13        # number of packets to generate
    time_t = 0
    print("\n--- Time {} {}, set simulation scenery".\
        format(time_t, TIME_UNIT)) 
    usreq = UserEquipment(None, v_vel=[1,2,0], debug=True)
    usreq.chan = Channel()
    usreq.chan_state = 100      # optimal channel state value
    usreq.tr_blk = TransportBlock()
    id_pref = "PQ-"
    usreq.pktque_dl = PacketQueue(id_pref, keep_pkts=keep_pkts)
    trfgen = TrafficGenerator(usreq.pktque_dl, 40, 1, nr_pkts) 
    # when run will generate 13 packets of 40 bits in a single time

    print("   ", usreq)
    print("       ", usreq.chan)
    print("       ", usreq.tr_blk)
    print("        Data queue:", usreq.pktque_dl)
    print("       ", trfgen)

    time_t += 1
    print("\n--- Time {} {}: generate {} packets of 40 bits".\
        format(time_t, TIME_UNIT, nr_pkts))
    trfgen.run(1)        # run traffic generator, insert in packet queue

    trfgen.gen_size, trfgen.nr_pkts = 60, 3
    trfgen.run(1)        # run traffic generator, insert in packet queue

    usreq.pktque_dl.show_counters()

    time_t += 1
    print("\n--- Time {} {}, create resources".format(time_t, TIME_UNIT)) 
    ls_res = [ Resource("Poor", [11, 2, 1]), Resource("Fair", [11, 4, 1]), \
             Resource("Good", [11, 8, 1]) ]
    for res in ls_res:
        print(res)
    while usreq.pktque_dl.ls_recvd or usreq.pktque_dl.dc_retrans:
        # received list or retransmission dict not empty
        time_t += 1
        one_user_rec_send(usreq, ls_res, time_t, loss_prob)

    print("\n--- End simulation, show counters")
    usreq.pktque_dl.show_counters()
    return(time_t)


def one_user_rec_send(usreq, ls_res, time_t, loss_prob):
    '''Receive and send packets, with a probability loss.

    @param usreq: a UserEquipment object.
    @param ls_res: a list of resources assigned to this user equipment.
    @param time_t: simulation instant time, for a timestamp.
    @param loss_prob: probability that a transport block is lost.
    @return: simulation instant time.
    '''
    print("\n--- Time {} {}, create 0,1,2 TBs, transmit".\
        format(time_t, TIME_UNIT)) 
    ls_tr_blk, tbs_total = usreq.mk_ls_trblk(ls_res, time_t)
    usreq.pktque_dl.show_pending()
    print("Transmission:")
    for tr_blk in ls_tr_blk:
        lost_sort = random()
        if lost_sort < loss_prob: 
            print("    Transport block {} lost".format(tr_blk[0]))
            usreq.pktque_dl.send_tb(tr_blk[0], "Lost", time_t)
        else:
            usreq.pktque_dl.send_tb(tr_blk[0], "Sent", time_t) 
            print("    Transport block {} sent".format(tr_blk[0]))
    usreq.pktque_dl.show_ls_pkts("Received")
    usreq.pktque_dl.show_pending()
    usreq.pktque_dl.show_retrans()
    usreq.pktque_dl.show_ls_pkts("Sent")
    return(time_t)


if __name__ == "__main__":
    print("Usage:  python3 qa_oneusreq.py [keep_pkts]")
    print("    where optional argument keep_pkts may be:")
    print("        y to keep all packets (default)")
    print("        n to discard all packets")
    print("        an integer, to keep the indicated number of packets")
    if len(sys.argv) > 1:
        keep_pkts = sys.argv[1]
        if keep_pkts == "y":
            keep_pkts = True
        elif keep_pkts == "n":
            keep_pkts = False
        else:
            keep_pkts = int(keep_pkts)
    else:
        keep_pkts = True
    one_user(keep_pkts)

