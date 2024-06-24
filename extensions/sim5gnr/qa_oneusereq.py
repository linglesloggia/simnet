# -*- coding: utf-8 -*-

#import classes drom 5g library

from basenode import Resource
from usernode import UserEquipment,TransportBlock

# import classes from the main library
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S
from libsimnet.pktqueue import PacketQueue

# import abstract classes to overwrite
from models.channel.randfixchan.randfix_chan import Channel
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator

# Python imports
from random import random




##############PARAMETERS####################################

band = "n258" 
"""String with used band for simulation. """
robustMCS=False
long_cp = False
ul = True
dl = True
ul_dl="DL"
mimo_mod ="SU"
nlayers=1

syms_slot=14
nr_slots=1 
nr_sbands=12
res_name="PRB"
snr =15

################ QoS ########################

n_connections_dl = 10
n_connections_ul = 0
packet_size_dl = 1000  #(bytes)
packet_size_ul = 0
pk_arrival_rate_dl = 200  #packets per second
pk_arrival_rate_ul = 0  #packets per second
reqAvailability = '' # or 'High'
reqdelay = 5 #ms

req_th_dl = packet_size_dl* pk_arrival_rate_dl * 8
req_th_ul = packet_size_ul* pk_arrival_rate_ul * 8
###########################################################




def one_user():
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
    usreq = UserEquipment(None)
    usreq.chan = Channel()
    usreq.chan_state = snr     # channel state value
    usreq.tr_blk = TransportBlock(band,robustMCS,ul_dl,mimo_mod,nlayers)
    id_pref = "PQ-"
    usreq.pktque_dl = PacketQueue(id_pref)
    trfgen = TrafficGenerator(usreq.pktque_dl, 40, 1, nr_pkts) 
    # when run will generate 13 packets of 40 bits in a single round

    print("   ", usreq)
    print("       ", usreq.chan)
    print("       ", usreq.tr_blk)
    print("        Data queue:", usreq.pktque_dl)
    print("       ", trfgen)

    time_t += 1
    print("\n--- Time {} {}: generate {} packets of 40 bits".\
        format(time_t, TIME_UNIT, nr_pkts))
    trfgen.run(1)        # run traffic generator, insert in packet queue
    usreq.pktque_dl.show_counters()

    time_t += 1
    print("\n--- Time {} {}, create resources".format(time_t, TIME_UNIT)) 
    ls_res = [ 
        Resource(res_name, res_pars=[syms_slot,nr_slots, nr_sbands,band,dl,ul,long_cp]), 
        Resource("PRB", res_pars=[syms_slot,nr_slots, nr_sbands,band,dl,ul,long_cp]) ]
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
    one_user()









