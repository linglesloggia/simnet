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
from models.transpblock.simpletrblk.simple_trblk import TransportBlock

# Python imports
import unittest



class UserEquipment2(UserEquipment):

    def mk_ls_trblk(self, ls_res, time_t, one_tb=False):
        '''With resources available, inserts packets into transport blocks.

        @param ls_res: list of resources assigned to this user equipment at a certain time.
        @param time_t: simulation instant time, for a timestamp.
        @param one_tb: if True makes one transport block with all resources, if False makes a transport block for each resource.
        @return: a list of transport blocks for this user equipment.
        '''
        t_chan = time_t * TIME_UNIT_S    # time in seconds
        self.v_pos = self.v_pos + self.v_vel * t_chan
        self.one_tb = one_tb

        ls_tr_blk = []         # list of transport blocks created for this user
        if self.one_tb:
            tb_size = 0
            for res in ls_res:     # for each resource on this user equipment
                nr_syms = res.get_symbols()              # get number of symbols
                tb_size += self.tr_blk.get_tb_size(nr_syms, self.chan_state) #  bits
            tr_blk = self.pkt_que.mk_trblk(tb_size)  # make transport block
            if tr_blk:    # transport block is not None (void)
                ls_tr_blk += [tr_blk]
        else:
            for res in ls_res:     # for each resource on this user equipment
                nr_syms = res.get_symbols()              # get number of symbols
                tb_size = self.tr_blk.get_tb_size(nr_syms, self.chan_state)  #  bits
                tr_blk = self.pkt_que.mk_trblk(tb_size)  # make transport block
                if tr_blk:    # transport block is not None (void)
                    ls_tr_blk += [tr_blk]
        if self.debug and ls_res:       # there are resources in list
            msg = "    TBs for UserEq {}:\n".format(self.id_object)
            for tb in ls_tr_blk:
                id_tb, pkts = tb[0], tb[1:]  # id_tb, pkt_1, pkt_2, ...
                msg += "        {} :".format(id_tb)
                for pkt in pkts:
                    msg += "\n            {}".format(pkt)
                #msg += "\n"
            mutex_prt(msg)
        return ls_tr_blk



class MakeTransportBlockTest(unittest.TestCase):

    def setUp(self):
        '''Set up a minimum simulation scenery.
        '''
        self.usreq = UserEquipment2(None) # user equipment, no user group
        self.usreq.chan_state = 0.5      # set channel for 4 bit per symbol
        self.usreq.tr_blk = TransportBlock()
        self.usreq.pkt_que = PacketQueue("PQ-")

        print("=== Setup scenery:")
        print("   {}, channel state {}".format(self.usreq, self.usreq.chan_state))
        print("       ", self.usreq.tr_blk)
        print("        Data queue:", self.usreq.pkt_que)
        return


    def testMakeTBList(self):
        '''Make a TB for each available resources if size allows.
        '''
        print("=== Make one TB for each resource.")
        for time_t in range(0,5):
            self.usreq.pkt_que.receive(40, time_t)
        #self.usreq.pkt_que.show_ls_pkts("Received")

        # a resource allows 2 packets
        print("--- A resource which allows 2 packets in a transport block")
        ls_res = [ Resource("Poor", 11, 2, 1) ]
        ls_tr_blk = self.usreq.mk_ls_trblk(ls_res, 1)
        print(ls_tr_blk)
        result =[['TB-0', ['PQ-1:Pkt-1', 0, 0.0, 40], ['PQ-1:Pkt-2', 1, 0.0, 40]]] 
        self.assertEqual(ls_tr_blk, result)

        # 2 resources makes 2 tBs
        print("--- 2 resources result in 2 transport blocks")
        ls_res = [ Resource("Poor", 11, 2, 1), Resource("Poor", 11, 2, 1) ]
        ls_tr_blk = self.usreq.mk_ls_trblk(ls_res, 2)
        print(ls_tr_blk)
        result = [['TB-1', ['PQ-1:Pkt-3', 2, 0.0, 40], ['PQ-1:Pkt-4', 3, 0.0, 40]],\
            ['TB-2', ['PQ-1:Pkt-5', 4, 0.0, 40]]]
        self.assertEqual(ls_tr_blk, result)

        # a resource too small for a packet, no transport block made
        print("--- resource not enough for a packet, no transport block made")
        ls_res = [ Resource("Poor", 11, 2, 1) ]
        self.usreq.pkt_que.receive(100, time_t)
        ls_tr_blk = self.usreq.mk_ls_trblk(ls_res, 3)
        print(ls_tr_blk)
        result = []
        self.assertEqual(ls_tr_blk, result)

        return


    def testMakeUniqueTB(self):
        '''Make only one TB of size allowe by all resources available.
        '''
        print("=== Make only one TB with all resources available.")
        for time_t in range(0,5):
            self.usreq.pkt_que.receive(40, time_t)
        #self.usreq.pkt_que.show_ls_pkts("Received")

        # resources allow 4 packets
        print("--- 2 resources, each allows 2 packets in a transport block")
        ls_res = [ Resource("Poor", 11, 2, 1), Resource("Poor", 11, 2, 1) ]
        #ls_tr_blk = self.usreq.transmit(ls_res, 1)
        ls_tr_blk = self.usreq.mk_ls_trblk(ls_res, 1, one_tb=True)
        print(ls_tr_blk)
        #result =[['TB-0', ['PQ-1:Pkt-1', 0, 0.0, 40], ['PQ-1:Pkt-2', 1, 0.0, 40]]] 
        result = [['TB-0', ['PQ-2:Pkt-1', 0, 0.0, 40], ['PQ-2:Pkt-2', 1, 0.0, 40],\
            ['PQ-2:Pkt-3', 2, 0.0, 40], ['PQ-2:Pkt-4', 3, 0.0, 40]]]
        self.assertEqual(ls_tr_blk, result)

        return


    def tearDown(self):
        #self.fib_elems = None
        self.usreq.tr_blk = None
        self.usreq.pkt_que = None
        self.usreq = None
        print ("--- tearDown executed!\n")
        return


if __name__ == "__main__": 
    unittest.main()
    


