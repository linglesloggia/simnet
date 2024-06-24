#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# randfixchan: overwrites for a random or fixed channel state
#

'''Channel model for a random or fixed value channel state.
'''

# import abstract classes to overwrite
from libsimnet.usernode import Channel, ChannelEnvironment

# Python imports
from random import random           # for the Channel class



class Channel(Channel):
    '''A simple channel, model overwrites Channel in libsimnet.
    '''

    def __init__(self, ch_env=None, loss_prob=0.0, chan_mode="Random", val_1=-100, val_2=100):
        '''Constructor.

        In this simple model, channel state may be a random number in (val_1, val_2) or a fixed number val_1, according to the value of the mode parameter, "Random" or "Fixed".
        @param ch_env: pointer to ChannelEnvironment object.
        @param loss_prob: probability that transmission failed.
        @param chan_mode: "Random" for a random value, "Fixed" for a fixed value.
        @param val_1: lower value for random, or a fixed value to return.
        @param val_2: upper valuer for random, irrelevant for fixed.
        '''
        super().__init__(ch_env, loss_prob)
        self.chan_mode = chan_mode
        '''Channel mode, Random or Fixed.'''
        self.val_1 = val_1
        '''Lower value for random, or a fixed value to return.'''
        self.val_2 = val_2
        '''Upper valuer for random, irrelevant for fixed.'''
        return


    def get_chan_state(self, time_t, v_pos=0.0, v_vel=0.0):
        '''Returns a measure of the channel state.

        @param time_t: simulation instant time, for a timestamp.
        @param v_pos: position vector.
        @param v_vel: velocity vector.
        @return: channel state, a number.
        '''
        if self.chan_mode == "Random":
            if self.val_1 != 0.0 or self.val_2 != 0:
                return random() * (self.val_2 - self.val_1) + self.val_1
            else:
                return random()
        elif self.chan_mode == "Fixed":   # return a fixed value, val_1
            return self.val_1
        else:
            print("get_chan_state, invalid mode")
            return None


    def __str__(self):
        '''For pretty printing.'''
        msg = "Channel rand_fix {:s}, loss prob {:5.3f}, mode {}: {} {}".\
            format(self.id_object, self.loss_prob, self.chan_mode, 
                self.val_1, self.val_2)
        return msg




class ChannelEnvironment(ChannelEnvironment):
    '''Channel environment.
    '''

    def __init__(self):
        '''Constructor.
        '''
        super().__init__()
        return



