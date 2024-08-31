#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# file_channel: a channel from file model
#

'''
Channel from file model. 
'''

# import classes from the main library
from libsimnet.usernode import Channel
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S

# import channel environment from channel from file model
from file_chenv import ChannelEnvironment

# Python imports
import csv
from random import random
import sys





class Channel(Channel):
    '''Channel model which reads channel states from a file.
    '''

    dc_chan_state = {}
    '''Dictionary of channel states for all user equipments.'''


    def __init__(self, ch_env=None, interpol=False, f_name=None):
        '''Constructor.

        @param ch_env: pointer to a ChannelEnviroment object.
        @param interpol: if True, interpolates.
        '''
        super().__init__()
        self.ch_env = ch_env    # pointer to a ChannelEnvironment object
        '''Channel environment object.'''
        self.interpol = interpol
        '''If True, interpolates.'''
        self.f_name = "snr_4ues_1000_5gUmi.csv" 
        '''File name of channel states for all user equipments.'''
        if f_name:
            self.f_name = f_name

        self.id_usreq = "UE-2"   # to be loaded on simulation setup
        '''User equipment to which this channel is attached.'''

        # make dictionary of channel states for all user equipments
        if not Channel.dc_chan_state:
            Channel.dc_chan_state = self.mk_dc_chan_state(f_name)
        # make dictionary of channel states for this user equipment
        #self.dc_t_states = self.mk_dc_t_states()
        self.dc_t_states = {}

        """
        self.t_ant = 0
        self.index = 0
        self.ls_state = None
        """
        return


    def mk_dc_chan_state(self, f_name):
        '''Makes a dictionary of channel states by user equipment.

        @param f_name: file name of a CSV file with channel states.
        '''
        print("--- Making dict for all UEs from file ", f_name)
        dc_usreq = {}
        fp = open(f_name, "r", encoding='utf8')
        reader = csv.reader(fp, delimiter=';', quotechar='"', \
            quoting=csv.QUOTE_ALL, lineterminator='\n')
        i = 0
        for row in reader:
            if row[0] in dc_usreq:
                dc_usreq[row[0]] = dc_usreq[row[0]] + [row[1:]]
            else:
                dc_usreq[row[0]] =  [row[1:] ]
            i += 1
            if i > 10:
                break
        fp.close()
        return dc_usreq


    def mk_dc_t_states(self):
        print("--- Making dict for {}".format(self.id_usreq))
        dc_usreq_state = {}
        if self.id_usreq in Channel.dc_chan_state:
            for item in Channel.dc_chan_state[self.id_usreq]:
                t_state = item[0]
                state, pos, vel = item[1].split()
                # convert to float and round to 3 decimals
                t_state = round(float(t_state), 3)
                state = round(float(state), 3)
                pos, vel = round(float(pos), 3), round(float(vel), 3) 
                # add to user equipment dictionary of states
                dc_usreq_state[t_state] = [state, pos, vel]
            return dc_usreq_state
        else:
            print("{}: no channel state for this user equipment".\
                format(self.id_usreq))
            return {}


    def get_chan_state(self, t, MS_pos, MS_vel):
        '''Return a measure of the channel state.

        Channel state is read from a file obtained from a channel simulator.
        @return: channel state, a number.
        '''

        # make dictionary of channel states for this user equipment
        if not self.dc_t_states:
            self.dc_t_states = self.mk_dc_t_states()
            #print(self.id_usreq, self.dc_t_states)

        t_secs = t * TIME_UNIT_S  # convert time to seconds
        t_secs = round(t_secs, 3)
        print("    time {}, t_secs {}".format(t, t_secs))


        if t_secs in self.dc_t_states:
            return self.dc_t_states[t_secs][0]
        else:
            print("No state for {}, time {}".format(self.id_usreq, t_secs))
            return 0

        """
        if self.ls_state is not None:
            if not self.interpol:
                if t<= self.ls_state[self.index][0]:
                    chan_state = self.ls_state[self.index][1][0]
                else:
                    for j in range(self.index+1,len(self.ls_state)):
                        if t <= self.ls_state[j][0] :
                            self.index = j
                            break
                    chan_state = self.ls_state[self.index][1][0]
            else:
                if t<= self.ls_state[self.index][0]:
                    if self.index == 0:  # extrapolation
                         y1 = self.ls_state[self.index][1][0]
                         t1 = self.ls_state[self.index][0]
                         y2 = self.ls_state[self.index+1][1][0]
                         t2 = self.ls_state[self.index+1][0]
                    else:
                         y1 = self.ls_state[self.index-1][1][0]
                         t1 = self.ls_state[self.index-1][0]
                         y2 = self.ls_state[self.index][1][0]
                         t2 = self.ls_state[self.index][0]
                else:
                    for j in range(self.index+1,len(self.ls_state)):
                        if t <= self.ls_state[j][0] :
                            self.index = j
                            break
                    y1 = self.ls_state[self.index-1][1][0]
                    t1 = self.ls_state[self.index-1][0]
                    y2 = self.ls_state[self.index][1][0]
                    t2 = self.ls_state[self.index][0]                
                chan_state = self.interpolate(t1, y1, t2, y2, t)
        else:
            print( "Warning : no file channel state for this user, return 0.")
            #chan_state = random()
            chan_state = 0
        return chan_state

    def interpolate(self, t1, y1, t2, y2, t):
        '''Interpolates time and position values.

        @param t1: time 1.
        @param y1: position 1.
        @param t2: time 2.
        @param y2: position 2.
        @param t: simulation time.
        @return: interpolated value.
        '''

        return (y2 - y1)/(t2 - t1)*(t - t1) + y1
        """
        

if __name__ == "__main__":
    f_name = "snr_4ues_1000_5gUmi.csv"
    ch_env_obj = ChannelEnvironment()
    chan_obj = Channel(ch_env_obj, True, f_name)

    #print("States for UE-2")
    #print(chan_obj.dc_t_states) #not yet built, until get_chan_state


    pos = [0,0,0]
    vel = [0,0,0]

    print("--- UE-2, states")
    for time_t in range(1,5):
        # t_chan = time_t * 0.00153
        #t_chan = time_t * 1.53  # time converted to seconds in get_chan_state
        t_chan = time_t
        chan_state = chan_obj.get_chan_state(t_chan, pos, vel)
        print("    channel time {}, state {}".format(t_chan, chan_state))

        #print("  time {:6.5f}, channel state: {:6.5f}".format(t_chan, chan_state) )

