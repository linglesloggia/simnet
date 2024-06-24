#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# pw_channel: PyWiCh channel model
#

'''
Simulator with PyWich channel model.
'''

# Math imports
from numpy import array

from libsimnet.usernode import Channel
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S


class Channel(Channel):
    ''' Customized for PyWich channel simulator.
    '''

    def __init__(self, ch_env=None, loss_prob=0.0):
        '''Constructor.
        '''
        super().__init__()
        self.ch_env = ch_env    # pointer to a ChannelEnvironment object


        return


    def get_chan_state(self, t, MS_pos, MS_vel):
        '''Return a measure of the channel state.

        Channel state is determined by the PyWich channel simulator.
        @return: channel state, a number.
        '''
        t = t*TIME_UNIT_S

        force_los = 2
        mode = 1
        chan_state, rxpsd, H, G, linear_losses, snr_pl, sp_eff, snr_pl_shadow = \
            self.ch_env.performance.compute_point(self.ch_env.fading, self.ch_env.freq_band, \
            self.ch_env.aBS, self.ch_env.aMS, MS_pos, MS_vel, t, force_los, mode)               
       
        return chan_state







