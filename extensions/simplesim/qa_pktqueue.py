#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_pktqueue : test functions for pktqueue
#

'''QA tests for the PacketQueue class.
'''

# import classes from the main library
from libsimnet.pktqueue import PacketQueue
from libsimnet.libutils import run_qa_tests

# Pyrhon imports
from time import perf_counter, sleep
import sys


def keep_pkts_op():
    '''Auxiliary function, asks whether to keep packets.
    '''
    op = input("    Keep packets(y, n, or integer): ")
    if op == "y":
        return True
    elif op == "n":
        return False
    else:
        try:
            return int(op)
        except:
            print("    ERROR: must be y, n or integer; discard packets.")
            return False


def len_bits_test():
    '''Tests length in bits function for different argument types.
    '''
    pkq = PacketQueue()
    ls_test = [ 13, "ABC", ["Pkt-0", 0.0, 0.0, 13], \
        ["Pkt-1", 0.0, 0.0, "0123456789"] ]
    ls_res = [ 13, 3*8, 13, 10*8 ]
    print("    Data   Function result    Correct result")
    for i in range(0, len(ls_test)):
        print("    {} : {:4d} : {:4d}".\
            format(ls_test[i], pkq.size_bits(ls_test[i]), ls_res[i] ) )
    return


def rec_send_tb():
    '''Receive and send packets using transport blocks.
    '''

    def nr_rec_send(time_t, rec_snd, q_obj):
        msg = "--- time {:d}: received {:d}, sent {:d}. {}:".\
            format(time_t, q_obj.dc_traf["Received"][0], \
                q_obj.dc_traf["Sent"][0], rec_snd)
        print(msg)
        return

    op = keep_pkts_op()
    q_obj = PacketQueue(keep_pkts=op, last_k=4)
    t_stamp = 1

    print("\n--- time {}: add 3 packets of 10 bits to data packet queue".\
        format(t_stamp))
    for i in range(0,3):
        q_obj.receive("DataPkt_" + str(q_obj.dc_traf["Received"][0]), t_stamp)
    q_obj.show_ls_pkts("Received")
    #q_obj.show_counters()
    t_stamp +=1

    print("\n--- time {}: Make a transport block of 200 bits".format(t_stamp))
    tr_blk = q_obj.mk_trblk(200, t_stamp)
    #q_obj.show_trblk(tr_blk)
    q_obj.show_pending()
    print("--  TB-0 lost, move packets to retransmission")
    q_obj.show_ls_pkts("Received")
    q_obj.send_tb("TB-0", "Lost", t_stamp)
    q_obj.show_pending()
    q_obj.show_retrans()
    t_stamp +=1

    print("\n--- time {}: add 4 packets of 12 bits to data packet queue".\
        format(t_stamp))
    for i in range(3,7):
        q_obj.receive("DataPkt_" + str(q_obj.dc_traf["Received"][0]), t_stamp)
    q_obj.show_ls_pkts("Received")
    t_stamp +=1

    print("\n--- time {}: add 6 packets of 12 bits to data packet queue".\
        format(t_stamp))
    for i in range(7,13):
        q_obj.receive("DataPkt_" + str(q_obj.dc_traf["Received"][0]), t_stamp)
    q_obj.show_ls_pkts("Received")
    t_stamp +=1

    print("\n--- time {}: Make transport blocks of 230, 100, and 340 bits".\
        format(t_stamp))
    tr_blk = q_obj.mk_trblk(230, t_stamp)
    tr_blk = q_obj.mk_trblk(100, t_stamp)
    tr_blk = q_obj.mk_trblk(340, t_stamp)
    #q_obj.show_trblk(tr_blk)
    q_obj.show_pending()
    q_obj.show_ls_pkts("Received")
    print("--  Packet TB-1 lost, TB-2 y TB-3 successfully sent")
    q_obj.send_tb("TB-1", "Lost", t_stamp)
    q_obj.send_tb("TB-2", "Sent", t_stamp)
    q_obj.send_tb("TB-3", "Sent", t_stamp)
    q_obj.show_pending()
    q_obj.show_retrans()
    q_obj.show_ls_pkts("Sent")
    t_stamp +=1

    print("\n--- time {}: make transport block of 230 bits".format(t_stamp))
    tr_blk = q_obj.mk_trblk(230, t_stamp)
    q_obj.show_trblk(tr_blk)
    q_obj.show_retrans()
    q_obj.show_pending()
    print("--  Packet TB-4 successfully sent")
    q_obj.send_tb("TB-4", "Sent", t_stamp)
    q_obj.show_ls_pkts("Received")
    q_obj.show_pending()
    q_obj.show_retrans()
    q_obj.show_ls_pkts("Sent")
    t_stamp +=1

    # show traffic counters
    q_obj.show_counters()
    q_obj.show_last_k()
    return
 

def rec_send_int(keep_pkts=True):
    '''Receives and sends packets as integers, shows counters.

    @param keep_pkts: whether to log packets or just counters, default True.
    '''
    op = keep_pkts_op()
    q_obj = PacketQueue(keep_pkts=op)
    t_stamp = 1         # to increment on each receive and send

    
    print("--- receive 1 packet as string")
    q_obj.receive("Pkt" + str(q_obj.dc_traf["Received"][0]), t_stamp)
    t_stamp += 1
    q_obj.show_ls_pkts("Received")
    q_obj.show_counters()

    print("--- receive 2 packets as int")
    for i in range(0,2):
        q_obj.receive(40, t_stamp)
        t_stamp += 1
    q_obj.show_ls_pkts("Received")
    q_obj.show_counters()

    print("--- transmit 4, 1 in excess")
    for i in range(0,4):  # one transmission in excess
        q_obj.send_pkt(t_stamp)
        t_stamp += 1
    q_obj.show_ls_pkts("Sent")
    q_obj.show_counters()
    return


def max_len_test(max_len=3):
    '''Tests maximum length of queue and drops packets.

    @param max_len: maximum length of queue, if surpassed lose packets.
    '''
    q_obj = PacketQueue(max_len=max_len)
    t_stamp = 1         # to increment on each receive and send
    print("--- receive 3 packets, max length={:d}".format(max_len))
    for i in range(0,3):
        q_obj.receive("Pkt" + str(q_obj.dc_traf["Received"][0]), t_stamp)
        sleep(0.01)
    q_obj.show_ls_pkts("Received")
    q_obj.show_counters()
    print("--- receive 2 packets more, max length={:d}".format(max_len))
    t_stamp += 1
    for i in range(0,2):
        q_obj.receive("Pkt" + str(q_obj.dc_traf["Received"][0]), t_stamp)
        sleep(0.01)
    q_obj.show_ls_pkts("Received")
    q_obj.show_counters()
    print("--- transmit 2 packets, receive 3 packets")
    t_stamp += 1
    for i in range(0,2):  # one transmission in excess
        q_obj.send_pkt()
        sleep(0.01)
    for i in range(0,3):
        q_obj.receive("Pkt" + str(q_obj.dc_traf["Received"][0]), t_stamp)
        sleep(0.01)
    q_obj.show_ls_pkts("Received")
    q_obj.show_ls_pkts("Sent")
    q_obj.show_counters()
    return


def last_k_test():
    '''Tests keep packets from last k time execution units.
    '''
    last_k = 6      # last k time units to keep packets
    k = input("    k times to keep packets, k>=0 (default 6): ")
    if k:
        last_k = int(k)
    time_sim = 15   # total simulation time
    q_obj = PacketQueue(keep_pkts=True, last_k=last_k)
    t_stamp = 2     # to increment on each receive and send
    print("--- receive send {} packets, keep for last k times {:d}".\
        format(time_sim, last_k))
    for i in range(0, time_sim, t_stamp):
        q_obj.receive("Pkt" + str(q_obj.dc_traf["Received"][0]), i)
    # make transport blocks and transmit successfully
    for i in range(0, time_sim, t_stamp):
        t_stamp = i + 1     # transmit after receive 
        tr_blk = q_obj.mk_trblk(32, t_stamp)
        q_obj.send_tb(tr_blk[0], "Sent", t_stamp)
    # show results
    q_obj.show_ls_pkts("Received")
    q_obj.show_ls_pkts("Sent")
    #print("last_k", time_sim)
    q_obj.show_last_k(time_sim=time_sim)
    return


def perf_test():
    '''Tests performance receiving and sending a high numbe of packets.
    '''
    print("Receive 1000 packets of 10 bits, transmit 900, measure time.")
    q_obj = PacketQueue()
    start_time = perf_counter()
    for i in range(0,1000):
        q_obj.receive(10)
    for i in range(0,900):  # one transmission in excess
        q_obj.send_pkt()
    #q_obj.show_ls_pkts("Received")
    q_obj.show_counters()
    end_time = perf_counter()
    print("Time elapsed: {:10.3f}".format(end_time - start_time) )
    return


# for pydoctor
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    # All these docstrings do not work, not all epydoc works:
    # https://pydoctor.readthedocs.io/en/latest/docformat/epytext.html 
    #'''@var ls_tests: list of tests to show in menu.'''
    ls_tests = [ \
        ("Length in bits for different argument types", len_bits_test), \
        ("Receive and send via transport blocks", rec_send_tb), \
        ("Receive and send, string and integer 40 bits", rec_send_int), \
        ("Maximum length of received queue", max_len_test), \
        ("Last k times keep received and sent packets", last_k_test), \
        ("Performance test", perf_test) \
        ]
    '''A list of tests to show in menu.'''
    run_qa_tests(ls_tests)

