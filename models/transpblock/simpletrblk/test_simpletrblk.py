#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_mktrblk : unittest for packing data packets in transport block

'''A test module for transport block making.

Values used in the tests:
    - Channel.chan_state = 0.5, so that 
    - TransporBlock.get_tb_size() : returns 4 bits per symbol;
    - Resource.get_symbols() : returns 1, so
    - Resource("Poor, 11, 2, 1) : number of symbols 11*2*1 = 22; 
    - transport block size = 4 bits per symbol * 22 symbols = 88 bits TB size.
Resource capacities:
    - Resource("Poor", 11, 2, 1) ] : 4 bits/sym * 11*2*1 symbols = 88
    - Resource("Fair", 11, 4, 1) ] : 4 bits/sym * 11*4*1 symbols = 176
    - Resource("Good", 11, 8, 1) ] : 4 bits/sym * 11*8*1 symbols = 352
'''


# import classes from the main library
from libsimnet.usernode import UserEquipment
from libsimnet.basenode import Resource
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S
from libsimnet.pktqueue import PacketQueue
from libsimnet.libutils import mutex_prt

# import concrete classes overwritten from abstract classes
from simple_trblk import TransportBlock

# Python imports
import unittest


class MakeTransportBlockTest(unittest.TestCase):

    def setUp(self):
        '''Set up a minimum simulation scenery.
        '''
        #self.usreq = UserEquipment2(None) # user equipment, no user group
        self.usreq = UserEquipment(None) # user equipment, no user group
        self.usreq.chan_state = 0        # set channel for 4 bit per symbol
        self.usreq.tr_blk = TransportBlock()
        self.usreq.pktque_dl = PacketQueue("PQ-")

        print("--- Setup scenery:")
        print("   {}, channel state {}".format(self.usreq, self.usreq.chan_state))
        print("       ", self.usreq.tr_blk)
        print("        Data queue:", self.usreq.pktque_dl)
        return


    def testMakeTBList(self):
        '''Make a TB for each available resources if size allows.
        '''
        print("=== Make one TB for each resource.")
        self.usreq.make_tb = "OneTBbyRes"
        for time_t in range(0,5):
            self.usreq.pktque_dl.receive(40, time_t)
        self.usreq.pktque_dl.show_ls_pkts("Received")

        # a resource allows 2 packets
        print("--- A resource which allows 2 packets in a transport block")
        ls_res = [ Resource(res_type="Poor", res_pars=[11, 2, 1]) ]
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1, ul_dl="DL")
        print("TBs total {}; TBs list:".format(tbs_total))
        print(ls_tr_blk)
        result =[ \
            ['TB_DL-0', \
                ['PQ-1:Pkt-1', 0, 0.0, 40], ['PQ-1:Pkt-2', 1, 0.0, 40]] ] 
        self.assertEqual(ls_tr_blk, result)

        # 2 resources makes 2 tBs
        print("--- 2 resources result in 2 transport blocks")
        ls_res = [ Resource("Poor", [11, 2, 1]), Resource("Poor", [11, 2, 1]) ]
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 2)
        print("TBs total {}; TBs list:".format(tbs_total))
        print(ls_tr_blk)
        result = [ \
            ['TB_DL-1', \
                ['PQ-1:Pkt-3', 2, 0.0, 40], ['PQ-1:Pkt-4', 3, 0.0, 40]], \
            ['TB_DL-2', \
                ['PQ-1:Pkt-5', 4, 0.0, 40]] ]
        self.assertEqual(ls_tr_blk, result)

        # a resource too small for a packet, no transport block made
        print("--- resource not enough for a packet, no transport block made")
        ls_res = [ Resource("Poor", [11, 2, 1]) ]
        self.usreq.pktque_dl.receive(100, time_t)
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 3)
        print("TBs total {}; TBs list:".format(tbs_total))
        print(ls_tr_blk)
        result = []
        self.assertEqual(ls_tr_blk, result)

        return


    def testMakeUniqueTB(self):
        '''Make only one TB of size allowed by all resources available.
        '''
        print("=== Make only one TB with all resources available.")
        self.usreq.make_tb = "OneTBallRes"
        for time_t in range(0,5):
            self.usreq.pktque_dl.receive(40, time_t)
        self.usreq.pktque_dl.show_ls_pkts("Received")

        # resources allow 4 packets
        print("--- 2 resources, each allows 2 packets, 4 in transport block")
        ls_res = [ Resource("Poor", [11, 2, 1]), Resource("Poor", [11, 2, 1]) ]
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1)
        print("TBs total {}; TBs list:".format(tbs_total))
        print(ls_tr_blk)
        result =[ ['TB_DL-0', \
                    ['PQ-1:Pkt-1', 0, 0.0, 40], \
                    ['PQ-1:Pkt-2', 1, 0.0, 40], \
                    ['PQ-1:Pkt-3', 2, 0.0, 40], \
                    ['PQ-1:Pkt-4', 3, 0.0, 40]]] 
        self.assertEqual(ls_tr_blk, result)

        return

    def testMakeTBbyNrRes(self):
        '''Make only one TB with nr_res identical resources.
        '''
        print("=== Make only one TB with a number of identical  resources.")
        self.usreq.make_tb = "TBbyNrRes"
        for time_t in range(0,5):
            self.usreq.pktque_dl.receive(40, time_t)
        self.usreq.pktque_dl.show_ls_pkts("Received")

        # resources allow 4 packets
        print("--- 2 resources, each allows 2 packets, 4 in transport block")
        ls_res = [ Resource("Poor", [11, 2, 1]), Resource("Poor", [11, 2, 1]) ]
        ls_tr_blk, tbs_total = self.usreq.mk_ls_trblk(ls_res, 1)
        print("TBs total {}; TBs list:".format(tbs_total))
        result =[ ['TB_DL-0', \
                    ['PQ-1:Pkt-1', 0, 0.0, 40], \
                    ['PQ-1:Pkt-2', 1, 0.0, 40], \
                    ['PQ-1:Pkt-3', 2, 0.0, 40], \
                    ['PQ-1:Pkt-4', 3, 0.0, 40]]] 
        self.assertEqual(ls_tr_blk, result)

        return




    def tearDown(self):
        #self.fib_elems = None
        self.usreq.tr_blk = None
        self.usreq.pktque_dl = None
        self.usreq = None
        TransportBlock.counter = 0
        PacketQueue.counter = 0
        print ("--- tearDown executed!\n")
        return


if __name__ == "__main__": 
    unittest.main()
    


