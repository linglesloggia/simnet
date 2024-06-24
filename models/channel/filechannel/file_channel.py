#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# file_channel: a channel from file model
#

'''
Channel from file model. 

Structure of channel states CSV file, in text format::

    "<id_usereq>";"<time>";"<channelstate value_1 value_2>" 

This channel model reads the CSV file and creates first a dictionary of channel states for all user equipments::

    {<id_usreq_>:<list of time and states in string format>}

Example: for user equipment 'UE-1'::

    [['0.001', '-43.53261264287159 0.0 0.0'], ['0.002', '-30.720416455949284 0.0 0.0']] 

This contains all states for each user equipment. This dictionary is a class attribute.

From this first dictionary this model creates a list of times and channel states for a user equipment::

    [<time>, <state>, <value_1>, <value_2>]}

Example: for the user equipment to which the channel is attached, for time 0.001 in seconds::

    [0.001, -25.872, 0.0, 0.0]

The list is accessed orderly in time by keeping track of the last instant simulated, for an efficient retrieval. If a simulation instant time value is not found in the list, the channel state is obtained by interpolating the nearest time values in the list. Since the simulation runs usually in milliseconds, the simulation time is first converted to time in seconds using the global variables TIME_UNIT, the name of the real time unit, and TIME_UNIT_S, the equivalence in seconds of the simulation time.
'''

# import classes from the main library
from libsimnet.usernode import Channel
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S


# Python imports
import csv
from random import random
import sys



class Channel(Channel):
    '''Channel model which reads channel states from a file.
    '''

    # class variables
    dc_chan_state = {}
    '''Dictionary of channel states for all user equipments.'''
 

    def __init__(self, ch_env=None, interpol=False, f_name=None, debug=False,loss_prob=0):
        '''Constructor.

        @param ch_env: pointer to a ChannelEnviroment object.
        @param interpol: if True, interpolates.
        @param f_name: file name of channel states for all user equipments.
        @param debug: if True prints messages.
        '''
        super().__init__(ch_env=ch_env,loss_prob=loss_prob)
        #self.ch_env = ch_env    # pointer to a ChannelEnvironment object
        #'''Channel environment object.'''
        self.interpol = interpol
        '''If True, interpolates.'''
        self.f_name = f_name 
        '''File name of channel states for all user equipments.'''
        self.debug = debug
        '''If True prints messages.'''
        self.id_usreq = ""
        '''UserEquipment to which this Channel is attached.'''
        # make dictionary of channel states for all user equipments
        if not Channel.dc_chan_state:
            Channel.dc_chan_state = self.mk_dc_chan_state(self.f_name)
    
        # for list of states by time
        self.index = 0
        '''Index of last position in channel state list.'''
        self.ls_state = []
        '''List of channel states by time.'''
        return


    def mk_dc_chan_state(self, f_name):
        '''Makes a dictionary of channel states by user equipment.

        @param f_name: file name of a CSV file with channel states.
        '''
        if self.debug:
            print("--- Making dict for all UEs from file ", f_name)
        dc_usreq = {}
        fp = open(f_name, "r", encoding='utf8')
        reader = csv.reader(fp, delimiter=';', quotechar='"', \
            quoting=csv.QUOTE_ALL, lineterminator='\n')
            
        for row in reader:
            if row[0] in dc_usreq:
                dc_usreq[row[0]] = dc_usreq[row[0]] + [row[1:]]
                #ch_st = float(row[2].split()[0])
            else:
                dc_usreq[row[0]] =  [row[1:] ]
        fp.close()
        return dc_usreq

 


    def get_chan_state(self, t, MS_pos, MS_vel):
        '''Return a measure of the channel state.

        Channel state is read from a file obtained from a channel simulator.
        @param MS_pos: position vector.
        @param MS_vel: velocity vector.
        @return: channel state, a number.
        '''
        t = t*TIME_UNIT_S
        if not self.ls_state:
            try:
                self.ls_state_aux = self.dc_chan_state[self.id_usreq]
                for row in self.ls_state_aux:
                    state = [float(ele) for ele in row[1].split()]
                    self.ls_state += [[float(row[0]), state]]
            except Exception as error:
                print("An exception occurred:", error)
                print( "Warning : no file channel state for this user, return 0.")
                return 0 
        if self.ls_state is not None:
            if not self.interpol:
                if t<= self.ls_state[self.index][0]:
                    chan_state = self.ls_state[self.index][1][0]
                else:
                    for j in range(self.index+1, len(self.ls_state)):
                        if t <= self.ls_state[j][0]:
                            self.index = j
                            break
                    chan_state = self.ls_state[self.index][1][0]
            else:
                if t <= self.ls_state[self.index][0]:
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
                    for j in range(self.index+1, len(self.ls_state)):
                        if t <= self.ls_state[j][0] :
                            self.index = j
                            break
                    y1 = self.ls_state[self.index-1][1][0]
                    t1 = self.ls_state[self.index-1][0]
                    y2 = self.ls_state[self.index][1][0]
                    t2 = self.ls_state[self.index][0]                
                if self.debug:
                    print("    Interpol {}: {:7.5f}:{:7.5f}, {:7.5f}:{:7.5f},{:7.5f},{}".\
                        format(self.index, t1, y1, t2, y2,t,self.id_usreq))
                chan_state = self.interpolate(t1, y1, t2, y2, t )
            return chan_state
        else:
            print( "Warning : no file channel state for this user, return 0.")
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

        intrplt = (y2 - y1)/(t2 - t1)*(t - t1) + y1
        #print("  interpol on {}: t1,y1={},{}; t2,y2={},{}; interpol {}".\
        #    format(t, t1, round(y1,3), t2, round(y2,3), round(intrplt,3)) )
        return intrplt


    def __str__(self):
        '''For pretty printing.'''
        msg = "Channel from file {:s}, usreq: {}".\
            format(self.f_name, self.id_usreq)
        return msg


if __name__ == "__main__":

    print("To run tests please do:")
    print("    python3 extensions/simplesim/qa_simulator.py")
