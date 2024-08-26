#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# pw_channel: PyWiCh channel model
#

'''
Simulator with PyWich channel model.
'''

# Math imports

from libsimnet.usernode import Channel
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S
import snr as sr

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

        mode = 2

        self.ch_env.fading.compute_ch_matrix(MS_pos,MS_vel,t=t,mode=mode)
        chan_state,rxPsd,H,G,ploss_linear,snr_pl,spectral_eff,snr_pl_shadow = sr.compute_snr(self.ch_env.fading,self.ch_env.freq_band)
        print("------chan state----",MS_pos,MS_vel,chan_state)
        return chan_state
