#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# test_TrafGenPoisson : unit test for Poissin traffic generator
#

'''Unit tests for the Poisson traffic generator.
'''

# import classes from the main library
from libsimnet.pktqueue import PacketQueue
from models.trafficgen.poissontrfgen.poisson_trfgen import TrafficGenerator


# Pyrhon imports
import unittest
import sys
#import numpy as np


class PoissonTrafficGenTest(unittest.TestCase):


    def setUp(self):
        '''Create a TrafficGenertor and a PacketQueue objects.
        '''
        self.gen_size = 10
        self.max_size = 1500
        pktque = PacketQueue()
        self.trfgen = TrafficGenerator(pktque, gen_size=self.gen_size, \
            max_size=self.max_size)
        return


    def test_run(self):
        '''Test traffic genertion in run() method.
        '''
        for i in range(0, 10):
            self.trfgen.run(i)
            last_pkt_size = self.trfgen.pktque.ls_recvd[-1][3]
            self.assertEqual(last_pkt_size, self.gen_size)
            self.assertIs(type(last_pkt_size), int)
        #self.trfgen.pktque.show_ls_pkts("Received") 
        self.trfgen.size_dist = "Exponential"
        for i in range(11, 1100):
            self.trfgen.run(i)
            last_pkt_size = self.trfgen.pktque.ls_recvd[-1][3]
            self.assertGreaterEqual(last_pkt_size, 0)
            self.assertLessEqual(last_pkt_size, self.max_size)
            self.assertIs(type(last_pkt_size), int)
            print(last_pkt_size, end=" ")
        print()
        #self.trfgen.pktque.show_ls_pkts("Received") 
        return


    def tearDown(self):
        '''Cleanup scenery.
        '''
        #PacketQueue.counter = 0     # to restart id in PQ-1
        return

if __name__ == "__main__": 
    unittest.main()


