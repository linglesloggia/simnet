#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# ut_usereq : unit test for UserEquipment
#

'''Unit tests for the UserEquipment classs.
'''

# import classes from the main library
from libsimnet.usernode import UserEquipment
from libsimnet.pktqueue import PacketQueue
from libsimnet.basenode import Resource
from models.channel.randfixchan.randfix_chan import Channel
from models.transpblock.simpletrblk.simple_trblk import TransportBlock

# Pyrhon imports
import unittest
import sys
import numpy as np

class UserEquipmentTest(unittest.TestCase):


    def setUp(self):
        '''Create a UserEquipment object.
        '''
        self.usreq = UserEquipment("NN", v_pos=[0,0,0], v_vel=[1,2,3])
        '''UserEquipment object.'''
        self.usreq.chan_state = 30     # channel state for 8 bits/symbol
        self.usreq.pktque_dl = PacketQueue()
        #self.usreq.chan = Channel(chan_mode="Fixed", val_1=100) 
        #print("Channel state", self.usreq.chan_state)
        self.usreq.tr_blk = TransportBlock()
        return


    def test_pos_move(self):
        '''Test position  and movement.
        '''
        self.assertEqual(list(self.usreq.v_pos), [0, 0, 0])
        self.assertEqual(list(self.usreq.v_vel), [1, 2, 3])
        self.usreq.pos_move(100)    # move v_vel * 100 ms
        self.assertEqual(list(self.usreq.v_pos), [0.1, 0.2, 0.3])
        #print(self.usreq)
        self.usreq.pos_move(300)    # move v_vel * 200 ms
        self.assertEqual(list(self.usreq.v_pos), [0.3, 0.6, 0.9])
        #print(self.usreq)
        self.usreq.pos_move(400, [4,5,6]) # move new v_vel * 100 ms
        self.assertEqual(list(self.usreq.v_pos), [0.7, 1.1, 1.5])
        self.assertEqual(list(self.usreq.v_vel), [4, 5, 6])
        #print(self.usreq)
        return


    def test_set_pos_vel(self):
        '''Tests setting of position and velocity vectors.
        '''
        v_pos_vrfy, v_vel_vrfy = [11, 12, 13], [0.1, 0.2, 0.3]
        self.usreq.set_pos_vel(v_pos_vrfy, v_vel_vrfy)
        v_pos, v_vel = self.usreq.get_pos_vel()
        self.assertListEqual(v_pos, v_pos_vrfy)
        self.assertListEqual(v_vel, v_vel_vrfy)
        return


    def test_mk_ls_trblk(self):
        '''Tests addition of packets to a queue.

        Tests:
            - gen_traffic: generate packets into the packet queue.
            - mk_ls_trblk: make transport blocks with resources available.
        '''
        for i in range(1, 5, 2):
            self.usreq.gen_traffic(40, i)                 # 40 bits as int
            self.usreq.gen_traffic("Pkt-"+str(i+1), i+1)  # 40 bits as string
        #print(self.usreq.pktque_dl.ls_recvd)
        self.assertEqual(self.usreq.pktque_dl.dc_traf["Received"], [4, 160, 40, 0])

        ls_res = []
        for i in range(0, 4):
            ls_res += [Resource("ResTest", [1, 5, 1])]  # 5 symbols*8 bits=40
            #print(ls_res[-1])
        # OneTBallRes, make one transport block with all resources
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1)
        #print("OneTBallRes", ls_tr_blk, tbs_total)
        self.assertEqual(len(ls_tr_blk[0][1]), 4)  # all packets in a TB
        self.assertEqual(tbs_total, 160)    # 4 TBs of 40 bits
        # OneTBbyRes, make one transport block for each resource
        for i in range(1, 5, 2):  # mk_ls_trblk extracted packets from queue 
            self.usreq.gen_traffic(40, i)                 # 40 bits as int
            self.usreq.gen_traffic("Pkt-"+str(i+1), i+1)  # 40 bits as string
        self.usreq.make_tb = "OneTBbyRes"
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1)
        #print("OneTBbyRes", ls_tr_blk, tbs_total)
        self.assertEqual(len(ls_tr_blk), 4)  # 4 TBs, one for each resource
        self.assertEqual(tbs_total, 160)     # 1 TB,  4 resources * 40 bits
        # TBbyNrRes, make one transport block by quantity of equal resources
        for i in range(1, 5, 2):  # mk_ls_trblk extracted packets from queue 
            self.usreq.gen_traffic(40, i)                 # 40 bits as int
            self.usreq.gen_traffic("Pkt-"+str(i+1), i+1)  # 40 bits as string
        self.usreq.make_tb = "TBbyNrRes"
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1)
        #print("TBbyNrRes", ls_tr_blk, tbs_total)
        self.assertEqual(len(ls_tr_blk[0][1]), 4)  # 4 TBs, one for each resource
        self.assertEqual(tbs_total, 160)     # 1 TB,  4 resources * 40 bits

        return


    def tearDown(self):
        '''Cleanup scenery.
        '''
        #PacketQueue.counter = 0     # to restart id in PQ-1
        return

if __name__ == "__main__": 
    unittest.main()


