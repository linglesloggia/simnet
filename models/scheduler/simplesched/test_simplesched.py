#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# ut_simple_sched : unittest for a simple scheduler model

'''Unit test for a simple scheduler.

Tests assignment of resources to a list of user equipments with their packet queues and traffic generators.
'''


# import classes from the main library
from libsimnet.usernode import UserEquipment
from libsimnet.basenode import Resource
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S
from libsimnet.pktqueue import PacketQueue
from libsimnet.libutils import mutex_prt

# import concrete classes overwritten from abstract classes
from models.transpblock.simpletrblk.simple_trblk import TransportBlock
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from simple_sched import Scheduler

# Python imports
import unittest
import sys

class SimpleSchedulerTest(unittest.TestCase):

    def setUp(self):
        '''Set up a minimum simulation scenery.
        '''
        self.ls_usreqs = []
        '''A list of user equipments.'''
        self.ls_res = []
        '''A list of resources.'''
        self.ls_trfgen = []
        '''A list of traffic generators.'''
        self.sched = Scheduler()
        '''The scheduler to assign resources to user equipments.'''
        self.debug = True

        print("=== SetUp scenery")
        # create list of UserEquipment objects
        nr_pkts = 3
        nr_users = 6
        nr_res = 10
        for i in range(0, nr_users):
            usreq = UserEquipment(None)
            usreq.pktque_dl = PacketQueue("PQ-")
            self.ls_usreqs += [usreq]
            trfgen = TrafficGenerator(usreq.pktque_dl, 40, 1, nr_pkts)
            if i in [2, 4]:  # no packets for these user equipments
                pass
            else:
                trfgen.run(0)     # adds packets to user equipment queue
            self.ls_trfgen += [trfgen]
        # create list of Resources
        for i in range(0, nr_res):
            self.ls_res += [ Resource("Poor", [11, 2, 1]) ]
        if self.debug:
            print("--- User equipments")
            for usreq in self.ls_usreqs:
                print("    {}, packets in queue {}".\
                    format(usreq, len(usreq.pktque_dl.ls_recvd)))
            print("--- Resources")
            for res in self.ls_res:
                print("    {}".format(res))
            print("--- Traffic generators")
            for trfgen in self.ls_trfgen:
                print("    {}".format(trfgen))
        return

    def assgn_res(self, t):
        ls_usr_res = self.sched.assign_res(self.ls_usreqs, self.ls_res, 0)
        if self.debug:
            print("--- time {}".format(t))
            for usr_res in ls_usr_res:
                usreq = usr_res[0]
                print("    User equipment: {}, in queue {}, resources {}".\
                    format(usreq.id_object, len(usreq.pktque_dl.ls_recvd), \
                        len(usr_res) -1))
                print("    ", end="")
                for res in usr_res[1:]:
                    print("  {}".format(res.id_object), end="")
                print()
        ls_nr_res = [3, 3, 0, 2, 0, 2]
        ls_lens = []
        for it in ls_usr_res:
            ls_lens += [len(it) - 1]   # first element is user equipment
        return ls_lens


    def testScheduler(self):
        '''Assigns resources to user equipments.
        '''
        ls_result = [ \
            [3, 3, 0, 2, 0, 2],
            [2, 2, 0, 3, 0, 3],
            [3, 3, 0, 2, 0, 2],
            [2, 2, 0, 3, 0, 3],
            [3, 3, 0, 2, 0, 2]
            ]
        ls_assign = []
        for t in range(0, 5):
            ls_assign += [self.assgn_res(t)]
        print()
        print("Assignment", ls_assign)
        print("Result    ", ls_result)
        self.assertEqual(ls_assign, ls_result)

        return


    def tearDown(self):
        self.ls_usreqs = None
        self.ls_res = None
        self.ls_trfgen = None
        self.sched = None
        print ("\n--- tearDown executed!\n")
        return

if __name__ == "__main__": 
    """
    st_obj = SimpleSchedulerTest()
    st_obj.setUp()
    sys.exit()
    """
    unittest.main()
    


