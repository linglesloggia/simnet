#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# ut_pktqueue : unit test for pktqueue
#

'''Unit tests for the PacketQueue class.
'''

# import classes from the main library
from libsimnet.pktqueue import PacketQueue
#from pktqueue import PacketQueue
from libsimnet.libutils import run_qa_tests

# Pyrhon imports
from time import perf_counter, sleep
import unittest
import sys

class PacketQueueTest(unittest.TestCase):


    def setUp(self):
        '''Set up a minimum simulation scenery.
        '''
        self.pkq = PacketQueue(keep_pkts=True)
        '''PacketQueue object, its id incremented on each test.'''
        self.t_stamp = 0
        '''A timestamp of reception.'''
        self.pkt_count = 0
        '''Received packets counter.'''
        #print ("\n--- setUp executed!\n")

        return


    def test_len_bits_test(self):
        '''Tests length in bits function for different argument types.
        '''
        ls_test = [ 13, "ABC", ["Pkt-0", 0.0, 0.0, 13], \
            ["Pkt-1", 0.0, 0.0, "0123456789"] ]
        ls_vrfy = [ 13, 3*8, 13, 10*8 ]
        #print("    Data   Function result    Correct result")
        ls_res = []
        for i in range(0, len(ls_test)):
            ls_res += [self.pkq.size_bits(ls_test[i])]
        self.assertEqual(ls_res, ls_vrfy)
        return


    def test_rec_send_tb(self):
        '''Tests reception of packets and size in bits of all received.
        '''
        bits_recvd = 0
        # receive 3 packets
        for i in range(0,3):
            self.t_stamp += 1
            self.pkt_count += 1
            self.pkq.receive("DataPkt_" + str(self.pkt_count), self.t_stamp)
        ls_vrfy = [
            ['PQ-1:Pkt-1', 1, 0.0, 'DataPkt_1'], 
            ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'], 
            ['PQ-1:Pkt-3', 3, 0.0, 'DataPkt_3'] ] 
        self.assertEqual(self.pkq.ls_recvd, ls_vrfy)
        # receive 6 more packets, total 9
        for i in range(0,6):
            self.t_stamp += 1
            self.pkt_count += 1
            self.pkq.receive("DataPkt_" + str(self.pkt_count), self.t_stamp)
        self.assertEqual(len(self.pkq.ls_recvd), 9)
        # count size in bits of all packets received
        for pkt in self.pkq.ls_recvd:
            bits_recvd += self.pkq.size_bits(pkt)
        self.assertEqual(bits_recvd, 9*9*8) # 9 packets, 9 bytes, 8 bits/byte
        # make a transport block of 200 bits, for 200/72 = 2 packets
        tr_blk = self.pkq.mk_trblk(200, self.t_stamp)
        tr_blk_vrfy = ['TB-0', 
                ['PQ-1:Pkt-1', 1, 0.0, 'DataPkt_1'], 
                ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'] ] 
        self.assertEqual(tr_blk, tr_blk_vrfy)
        dc_pend_vrfy = {tr_blk_vrfy[0]:tr_blk_vrfy[1:]}
        self.assertEqual(self.pkq.dc_pend, dc_pend_vrfy)
        # TB-0 lost, move to retransmission
        self.pkq.send_tb("TB-0", "Lost", self.t_stamp)
        dc_retrans_vrfy = {
            'PQ-1:Pkt-1': ['PQ-1:Pkt-1', 1, 0.0, 'DataPkt_1'],
            'PQ-1:Pkt-2': ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'] } 
        self.assertEqual(self.pkq.dc_retrans, dc_retrans_vrfy)
        self.t_stamp += 1
        # make a transport block of 200 bits, for 216/72 = 3 packets
        tr_blk = self.pkq.mk_trblk(216, self.t_stamp)
        tr_blk_vrfy = ['TB-1',
            ['PQ-1:Pkt-1', 1, 0.0, 'DataPkt_1'],
            ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'],
            ['PQ-1:Pkt-3', 3, 0.0, 'DataPkt_3'] ]
        self.assertEqual(tr_blk, tr_blk_vrfy)
        # TB-1 lost, move additional packet to retransmission
        self.pkq.send_tb("TB-1", "Lost", self.t_stamp)
        dc_retrans_vrfy = {
            'PQ-1:Pkt-1': ['PQ-1:Pkt-1', 1, 0.0, 'DataPkt_1'],
            'PQ-1:Pkt-2': ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'],
            'PQ-1:Pkt-3': ['PQ-1:Pkt-3', 3, 0.0, 'DataPkt_3'] } 
        self.assertEqual(self.pkq.dc_retrans, dc_retrans_vrfy)
        # make a transport block of 72 bits, for 72/72 = 1 packets
        tr_blk = self.pkq.mk_trblk(72, self.t_stamp)
        # TB-2 successfully sent
        self.pkq.send_tb("TB-2", "Sent", self.t_stamp)
        dc_retrans_vrfy = {
            'PQ-1:Pkt-2': ['PQ-1:Pkt-2', 2, 0.0, 'DataPkt_2'], \
            'PQ-1:Pkt-3': ['PQ-1:Pkt-3', 3, 0.0, 'DataPkt_3'] } 
        self.assertEqual(self.pkq.dc_retrans, dc_retrans_vrfy)
        self.t_stamp += 1
        # make a transport block of 360 bits, for 360/72 = 5 packets
        tr_blk = self.pkq.mk_trblk(360, self.t_stamp)
        # TB-3 successfully sent
        self.pkq.send_tb("TB-3", "Sent", self.t_stamp)
        dc_retrans_vrfy = {}
        self.assertEqual(self.pkq.dc_retrans, dc_retrans_vrfy)
        ls_sent_vrfy = [
            ['PQ-1:Pkt-1', 1, 10, 'DataPkt_1'], 
            ['PQ-1:Pkt-2', 2, 11, 'DataPkt_2'], 
            ['PQ-1:Pkt-3', 3, 11, 'DataPkt_3'], 
            ['PQ-1:Pkt-4', 4, 11, 'DataPkt_4'], 
            ['PQ-1:Pkt-5', 5, 11, 'DataPkt_5'], 
            ['PQ-1:Pkt-6', 6, 11, 'DataPkt_6'] ] 
        self.assertEqual(self.pkq.ls_sent, ls_sent_vrfy)
        return


    def test_max_len(self, nr_gen=5, max_len=3):
        '''Tests maximum length of queue and drop packets.

        @param nr_gen: number of packets to generate.
        @param max_len: maximum length of queue, if surpassed lose packets.
        '''

        pkq = PacketQueue(max_len=max_len)
        for i in range(0, max_len):
            pkq.receive("Pkt-" + str(i), t_stamp=i)
        self.assertEqual(len(pkq.ls_recvd), max_len)
        return


    def test_last_k(self, time_sim=13, last_k=6):
        '''Tests keep packets from last k time execution units.
        '''
        pkq = PacketQueue(last_k=last_k)
        # receive packets
        for i in range(1, time_sim+1, 2):
            pkq.receive("Pkt-" + str(i), t_stamp=i)
        #print("Received:", pkq.ls_recvd)
        #print("Last {} received: {}".format(last_k, pkq.last_lst_rec))
        # transmit packets
        self.assertEqual(len(pkq.last_lst_rec), last_k/2)
        for i in range(1, time_sim+1, 2):
            # make a transport block of 48 bits, Pkt-00 6*8 = 48 bits
            tr_blk = pkq.mk_trblk(48, i+1)
            #print(i, tr_blk)
            # TB successfully sent, t_stamp i+1
            pkq.send_tb(tr_blk[0], "Sent", i+1)
        #print("Sent:", pkq.ls_sent)
        #print("Last {} sent: {}".format(last_k, pkq.last_lst_snt))
        self.assertEqual(len(pkq.last_lst_snt), last_k/2)
        # compare times of oldest itemns in last k items lists
        time_t_k = time_sim - last_k
        #print("Rec", time_t_k, pkq.last_lst_rec[0][0])
        #print("Snt", time_t_k, pkq.last_lst_snt[0][0])
        self.assertLessEqual(time_t_k, pkq.last_lst_rec[0][0])
        self.assertLessEqual(time_t_k, pkq.last_lst_snt[0][0])
        #pkq.show_last_k()
        return    

    def tearDown(self):
        self.pkg = None
        PacketQueue.counter = 0     # to restart id in PQ-1
        return

if __name__ == "__main__": 
    unittest.main()
    


