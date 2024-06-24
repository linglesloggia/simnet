#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simple_trblk: overwrites for a simple transport block model
#

'''Simple transport block model.
'''

# import abstract classes to overwrite
from libsimnet.usernode import TransportBlock



class TransportBlock(TransportBlock):
    '''Defines the number of data bits to include in a transport block.
    '''

    def __init__(self, max_size=0, min_size=0):
        '''Constructor.

        @param max_size: maximum size.
        @param min_size: minimum size.
        '''
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        return


    def get_tb_size(self, nr_syms, chan_state, nr_res=1, ul_dl="DL"):
        '''Determines number of bits to include in a transport block.

        Determines bits per symbol according to channel state, multiplies by number of symbols allowed to obtain number of bits allowed in a transport block.

        In this example, bits per symbol are 0, 1, 2, 4 or 8 for channel state in [-100, -60], [-60, -20], [-20, +20], [+20, +60], [+60, +100], respectively.
        @param nr_syms: number of symbols a resource allows.
        @param chan_state: channel state.
        @param nr_res: number of resources assigned.
        @param ul_dl: determine size for upload (UL) or download (DL) traffic.
        @return: transport block size in bits.
        '''
        if chan_state < -60:      # worst state of the channel
            bits_per_sym = 0
        elif chan_state < -20:
            bits_per_sym = 1
        elif chan_state < 20:
            bits_per_sym = 4
        elif chan_state < 60:
            bits_per_sym = 8
        else:               
            bits_per_sym = 16      # best state of the channel
        tb_size = nr_syms * bits_per_sym * nr_res
        return tb_size


    def __str__(self):
        '''For pretty printing.
        '''

        msg = "Transport block simple "
        msg += "{:s}, overhead {:d}, size min {:d}, max {:d}".\
            format(self.id_object, self.overhead, self.min_size, self.max_size)
        return msg

