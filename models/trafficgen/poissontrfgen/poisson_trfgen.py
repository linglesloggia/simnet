# -*- coding: utf-8 -*-

'''A Poisson traffic generator, overwrites TrafficGenerator in libsimnet.
'''

# import from the main library, abstract classes to overwrite
from libsimnet.usernode import TrafficGenerator
from libsimnet.basenode import TIME_UNIT

# Python imports
from random import random   # for the Channel class
from numpy import log       # for channel position and velocity

class TrafficGenerator(TrafficGenerator):
    '''Generates traffic on each user of a list of usreqs.

    A Poisson traffic generator with fixed packet size, exponential delay distribution between arrivals and a fixed number of packets on each instance of execution.
    '''


    def __init__(self, pktque, gen_size=10, gen_delay=1, nr_pkts=1, priority=1, \
            size_dist="Fixed", debug=False,max_size =1500):
        '''Constructor.
        
        @param pktque: packet queue in which to insert generated packets. 
        @param gen_size: average size in bits of data to include in packets to generate.
        @param gen_delay: average delay between interarrivals.
        @param nr_pkts: number of packets to generate on each instant time.
        @param priority: order of execution.
        @param size_dist: size of packet, may be "Fixed" or "Exponencial".
        @param debug: if True prints messages.
        '''
        super().__init__(pktque, priority, ul_dl="DL", debug=debug)
        self.gen_size = gen_size
        '''Size in bits of data to include in packets to generate.'''
        self.gen_delay = gen_delay
        '''Number of simulation intervals between generation.'''
        self.nr_pkts = nr_pkts
        '''Number of packets to generate on each instant time.'''
        self.size_dist = size_dist
        '''Size of packet, may be "Fixed" or "Exponencial".'''
        self.max_size = max_size
        return


    def get_nr_pkts(self, time_t):
        '''Determine number of packets to generate.

        @return: number of packets to generate.
        '''
        return self.nr_pkts


    def get_gen_delay(self, time_t):
        '''Determine average delay for next traffic generator event.

        @return: average delay for next traffic generator event.
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
        size = 0
        if self.debug:
            print("    Traffic Gen {}: packets {}, time {} {}".\
                format(self.id_object, nr_pkts, round(time_t, 3), TIME_UNIT))
        for i in range(0, nr_pkts):
            if self.size_dist == "Fixed":
                self.pktque.receive(gen_size, time_t)
                size = gen_size
            elif self.size_dist == "Exponential":
                psize = random()
                size = int(-log(1.0 - psize)*gen_size)
                if size < 1:
                    size = 1
                if size > self.max_size:
                    size = self.max_size
                self.pktque.receive(int(size), time_t)
            else:
                print("Invalid distribution size:", self.size_dist)
            if self.debug:
                print("    Traffic Gen {}: size {}, dist size {}".\
                format(self.id_object, size, self.size_dist))
        iat = self.inter_arrival_time(1/gen_delay)
        next_event = [time_t + iat, self.priority, self.id_object]
        return next_event


    def inter_arrival_time(self, rateParameter):
        '''Calculate inter arrival time.

        @param rateParameter: rate parameter.
        '''
        p = random()
        intarr_time = -log(1.0 - p)/rateParameter
        if self.debug:
            print("    Traffic Gen {}: interarrival time {}".\
                format(self.id_object, round(intarr_time, 3)))
        return intarr_time -log(1.0 - p)/rateParameter


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "TrafficGenerator Poisson "
        msg += "{:s}, gen size {:d}, delay {}, pkts {}".\
                format(self.id_object, self.gen_size, self.gen_delay, \
                    self.nr_pkts)
        return msg



