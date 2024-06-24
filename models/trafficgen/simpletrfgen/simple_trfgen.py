#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simple_trfgen: overwrite for a simple traffic generator
#

'''A simple traffic generator, overwrites TrafficGenerator in libsimnet.
'''

# import abstract classes to overwrite
from libsimnet.usernode import TrafficGenerator

from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S

# Python imports
from random import random           # for the Channel class



class TrafficGenerator(TrafficGenerator):
    '''Generates traffic on each user of a list of usreqs.
    
    A simple traffic generator with fixed packet size, fixed delay between packet generation, and a fixed number of packets on each instance of execution.
    '''

    def __init__(self, pktque, gen_size=10, gen_delay=1, nr_pkts=1, priority=1, \
            ul_dl="DL", debug=False):
        '''Constructor.

        @param pktque: packet queue in which to insert generated packets. 
        @param gen_size: size in bits of data to include in packets to generate.
        @param gen_delay: number of simulation time units between generation.
        @param nr_pkts: number of packets to generate on each instant time.
        @param priority: order of execution.
        @param ul_dl: traffic to generate, download (DL) or upload (UL).
        @param debug: if True prints messages.
        '''
        super().__init__(pktque, priority=priority, ul_dl=ul_dl, debug=debug)
        
        self.gen_size = gen_size
        '''Size in bits of data to include in packets to generate.'''
        self.gen_delay = gen_delay
        '''Number of simulation intervlas between generation.'''
        self.nr_pkts = nr_pkts
        '''Number of packets to generate on each instant time.'''

        return


    def get_nr_pkts(self, time_t):
        '''Determine number of packets to generate.

        @return: number of packets to generate.
        '''
        return self.nr_pkts


    def get_gen_delay(self, time_t):
        '''Determine delay for next traffic generator event.

        @return: delay for next traffic generator event.
        '''
        return self.gen_delay


    def get_gen_size(self, time_t):
        '''Determines size of packets to generate.

        @return: size of packets to generate.
        '''
        return self.gen_size


    def run(self, time_t):
        '''Generates traffic, returns new event.

        Traffic are data packets represented as a number of bits.
        @param time_t: simulation instant time.
        @return: next traffic generator event.
        '''

        nr_pkts = self.get_nr_pkts(time_t)
        gen_delay = self.get_gen_delay(time_t)
        gen_size = self.get_gen_size(time_t)
        if self.debug:
            print("    Traffic Gen {} : packets {}, time {} {}".\
                format(self.id_object, nr_pkts, time_t, TIME_UNIT))
        for i in range(0, nr_pkts):
            self.pktque.receive(gen_size, time_t)
        #next_event = [time_t + gen_delay, self.priority, self.id_object]
        next_event = [time_t + gen_delay, self.priority, self.id_object]
        return next_event


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "TrafficGenerator simple "
        msg += "{:s}, gen size {:d}, delay {}, pkts {}".\
                format(self.id_object, self.gen_size, self.gen_delay, \
                    self.nr_pkts)
        return msg



