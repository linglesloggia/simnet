#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# usernode.py: PyWiSim user equipment related entities module.
#

'''
PyWiSim, user equipment related entities module.
'''

# libsimnet imports
from libsimnet.libutils import mutex_prt
from libsimnet.basenode import TIME_UNIT, TIME_UNIT_S

# Python imports
from random import random
from time import sleep
from abc import ABC, abstractmethod
import numpy as np     # for channel position and velocity


class UserEquipment:
    '''Represents a user data terminal equipment (DTE) and its traffic.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, usr_grp, v_pos=[0.0,0.0,0.0], v_vel=[0.0,0.0,0.0], \
            make_tb="OneTBallRes", debug=False):
        '''Constructor.

        @param usr_grp: pointer to the UserGroup object to which this UserEquipment belongs.
        @param v_pos: three dimensional vector of position, a list.
        @param v_vel: three dimensional vector of velocity, a list.
        @param make_tb: "OneTBallRes" makes one TB with all resources, "OneTBbyRes" makes one TB for each resource, "TBbyNrRes" makes one TB with a number of resources of the same type.
        @param debug: if True print messages, if "3" prints more detailed messages.
        '''

        self.usr_grp = usr_grp
        '''User group to which this user equipment belongs.'''
        self.v_pos = np.array(v_pos)
        '''User equipment position vector, convert to numpy array type.'''
        self.v_vel = np.array(v_vel)
        '''User equipment velocity vector, convert to numpy array type.'''
        self.make_tb = make_tb
        '''Transport block modes according to resources.'''
        self.debug = debug
        '''If True prints debug messages, more detailed if value is 3.'''
        
        UserEquipment.counter += 1      # increment counter of objects created
        self.id_object = "UE-" + str(UserEquipment.counter)
        '''A unique object identifier.'''
        self.pktque_dl = None
        '''Packet queue to handle this user equiment download traffict.'''
        self.pktque_ul = None
        '''Packet queue to handle this user equipment upload traffic.'''
        self.chan = None
        '''Channel assigned to this user equipment.'''
        self.chan_state = None
        '''Channel state in a certain instant.'''
        self.tr_blk = None
        '''Transport block object, determines number of bits to transmit.'''
        self.trf_gen = None
        '''Traffic generator object, adds packets to user equipment queue.'''
        self.last_move_t = 0
        '''Time of last move action.'''

        return


    def pos_move(self, time_t, v_vel=None):
        '''Changes vector position according to a velocity vector.

        @param time_t: instant time.
        @param v_vel: velocity vector in m/s, or None to use self velocity attribute. If this parameter is not None, self velocity vector is updated to this value.
        '''
        if v_vel:
            self.v_vel = np.array(v_vel)  # updates local velocity
        delta_t = (time_t - self.last_move_t) * TIME_UNIT_S  # from last move
        self.v_pos = np.round(self.v_pos + self.v_vel * delta_t, 3)
        self.last_move_t = time_t         # update time of last move action
        return


    def get_pos_vel(self):
        '''Returns vector of position and velocity.

        @return: position vector, velocity vector.
        '''
        return self.v_pos, self.v_vel


    def mk_ls_trblk(self, ls_res, time_t, ul_dl="DL"):
        '''With resources available, inserts packets into transport blocks.

        @param ls_res: list of resources assigned to this user equipment at a certain time.
        @param time_t: simulation instant time, for a timestamp.
        @param ul_dl: make transport block and transmit from upload (UL) or download (DL) queue.
        @return: a list of transport blocks for this user equipment, and the total bits included in all transport blocks.
        '''

        def mk_tb_uldl(ul_dl, tb_size):
            '''Make transport block according to DL or UL packet queue.

            @param tb_size: size of transport block to build.
            @return: a transport block.
            '''
            if ul_dl == "DL":
                tr_blk = self.pktque_dl.mk_trblk(tb_size, time_t, \
                    tb_prefix="TB_DL-")
                #print("TB DL", tr_blk)
            elif ul_dl == "UL":
                tr_blk = self.pktque_ul.mk_trblk(tb_size, time_t, \
                    tb_prefix="TB_UL-")
                #print("TB UL", tr_blk)
            else:
                print("mk_ls_trblk, ERROR in ul_dl: {}".format(ul_dl)) 
                return None
            return tr_blk

        tbs_total = 0       # total bits included in transport blocks
        ls_tr_blk = []      # list of transport blocks created for this user
        if not ls_res:      # no resources available
            return ls_tr_blk, tbs_total
        if self.make_tb == "OneTBallRes":
            tb_size = 0
            for res in ls_res:     # for each resource on this user equipment
                if ul_dl != res.ul_dl:  # skip if resource not adequate
                    continue
                nr_syms = res.get_symbols(ul_dl)    # get number of symbols
                tb_size += self.tr_blk.get_tb_size(nr_syms, self.chan_state, \
                    ul_dl=ul_dl) #  bits to include in a transport block
            tbs_total = tb_size
            tr_blk = mk_tb_uldl(ul_dl, tb_size)  # make DL/UL transport block
            if tr_blk:    # transport block is not None (void)
                ls_tr_blk += [tr_blk]
        elif self.make_tb == "OneTBbyRes":
            for res in ls_res:     # for each resource on this user equipment
                if ul_dl != res.ul_dl:  # skip if resource not adequate
                    continue
                nr_syms = res.get_symbols(ul_dl)    # get number of symbols
                tb_size = self.tr_blk.get_tb_size(nr_syms, self.chan_state, \
                    ul_dl=ul_dl)
                tbs_total += tb_size
                tr_blk = mk_tb_uldl(ul_dl, tb_size)  # DL/UL transport block
                if tr_blk:    # transport block is not None (void)
                    ls_tr_blk += [tr_blk]
        elif self.make_tb == "TBbyNrRes":
            # assumes all resources are equal and all for UL or DL
            # get nr_syms from one resource
            nr_syms = ls_res[0].get_symbols(ul_dl)  # get number of symbols
            tb_size = self.tr_blk.get_tb_size(nr_syms, self.chan_state, \
                nr_res=len(ls_res), ul_dl=ul_dl)  # size with same type resources
            tbs_total = tb_size
            tr_blk = mk_tb_uldl(ul_dl, tb_size)  # make DL/UL transport block
            if tr_blk:    # transport block is not None (void)
                ls_tr_blk += [tr_blk]
        else:
            print("UserEquipment.make_tb error:", self.make_tb)
            return None
        if (self.debug == True or self.debug == 3) and ls_res:
            # there are resources in list
            nr_unsent = self.pktque_dl.dc_traf["Received"][0] - \
                self.pktque_dl.dc_traf["Sent"][0]
            msg = "    Resources for UserEq {}: {}; packets to send: {}".\
                format(self.id_object, len(ls_res), nr_unsent)
            if self.debug == 3:
                for res in ls_res:
                    msg += "\n        {}".format(res)
            mutex_prt(msg)
            msg = "    TBs for UserEq {}:\n".format(self.id_object)
            for tb in ls_tr_blk:
                id_tb, pkts = tb[0], tb[1:]  # id_tb, pkt_1, pkt_2, ...
                msg += "        {} :".format(id_tb)
                for pkt in pkts:
                    msg += "\n            {}".format(pkt)
                #msg += "\n"
            mutex_prt(msg)
        return ls_tr_blk, tbs_total

    
    def gen_traffic(self, nr_bits, time_t, ul_dl="DL"):
        '''Adds bits to send as a "packet" into the packet queue.

        Though traffic is usually received from a traffic generator, this function allows to insert data packets manually, mainly for testing purposes.
        @param nr_bits: number of bits in packet.
        @param time_t: simulation instant time, for a timestamp.
        @param ul_dl: traffic to generate, download (DL) or upload (UL).
        '''
        if ul_dl == "UL" and self.pktque_ul != None:  # if UL queue exists
            self.pktque_ul.receive(nr_bits, time_t)   # adds bits to UL queue
        else:  # assumes DL by default              
            self.pktque_dl.receive(nr_bits, time_t)   # adds bits to DL queue
        if self.debug:
            mutex_prt("..." + self.id_object + " adding " + str(nr_bits))
        return


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "UserEq {}, v_pos={}, v_vel={}".\
            format(self.id_object, self.v_pos, self.v_vel)
        return msg



class ChannelEnvironment(ABC):
    '''Channel environment.
    '''

    def __str__(self):
        '''For pretty printing.
        '''
        return "ChannelEnvironment"



class Channel(ABC):
    '''Communications link from user equipment to a slice in a base station.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''

    def __init__(self, ch_env=None, loss_prob=0.0):
        '''Constructor.

        @param ch_env: pointer to ChannelEnvironment object.
        @param loss_prob: probability that transmission failed.
        '''
        Channel.counter += 1
        self.id_object = "CH-" + str(Channel.counter)
        '''A unique object identifier.'''
        self.ch_env = ch_env    # pointer to a ChannelEnvironment object
        '''Pointer to Channel Environment object.'''
        self.loss_prob = loss_prob
        '''Probability that transmission failed.'''
        return


    @abstractmethod
    def get_chan_state(self):
        '''Abstract method, returns a measure of the channel state.
        '''
        pass


    def get_sent(self, loss_prob=None):
        '''Determines if transmission was successful or not.

        Transmission is successful if its probability is greater than the probability of being lost.
        @param loss_prob: probability that transmission failed, i.e. if less than this number transmission failed, otherwise it was successful.
        @return: True if transmission was successful, False otherwise.
        '''
        loss_prob = loss_prob if loss_prob else self.loss_prob
        loss_value = random()        
        if loss_value < loss_prob:
            return False
        else:
            return True


    def __str__(self):
        '''For pretty printing.'''
        msg = "Channel {:s}, loss prob {:5.3f}".\
            format(self.id_object, self.loss_prob)
        return msg



class TransportBlock(ABC):
    '''Defines the number of data bits to include in a transport block.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''

    def __init__(self):
        '''Constructor.
        '''

        TransportBlock.counter += 1
        self.id_object = "TB-" + str(TransportBlock.counter)
        '''A unique object identifier.'''
        
        # transport block properties
        self.overhead = 1
        '''Overhead.'''
        self.min_size = 1
        '''Minimum transport block size.'''
        self.max_size = 1
        '''Maximum transport block size.'''

        return


    @abstractmethod
    def get_tb_size(self):
        '''Determines number of bits to include in a transport block.
        '''
        pass
        return


    def __str__(self):
        '''For pretty printing.
        '''

        msg = "Transport block {:s}, overhead {:d}, size min {:d}, max {:d}".\
            format(self.id_object, self.overhead, self.min_size, self.max_size)
        return msg



class TrafficGenerator(ABC):
    '''Generates traffic on each user of a list of usreqs.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, pktque, priority=1, ul_dl="DL", debug=False):
        '''Constructor.

        @param pktque: packet queue in which to insert generated packets. 
        @param priority: order of execution.
        @param ul_dl: traffic to generate, download (DL) or upload (UL).
        @param debug: if True prints messages.
        '''
        
        self.pktque = pktque
        '''The PacketQueue object in which to insert generated packets.'''
        self.priority = 1 #priority
        '''Order of execution in a certain instant time.'''
        self.ul_dl = ul_dl
        '''Generate download (DL) or upload (UL) traffic.'''
        self.debug = debug
        '''If True prints debug messages.'''

        TrafficGenerator.counter += 1
        self.id_object = "TG-" + str(TrafficGenerator.counter)
        '''Unique object identifier.'''

        return


    @abstractmethod
    def get_nr_pkts(self):
        '''Determines number of packets to generate.
        '''
        pass
        return


    @abstractmethod
    def get_gen_delay(self, time_t):
        '''Determines delay for next traffic generator event.
        '''
        pass
        return


    @abstractmethod
    def get_gen_size(self, time_t):
        '''Determines size of packets to generate.
        '''
        pass
        return


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
        next_event = [time_t + self.gen_delay, self.priority, self.id_object]
        return next_event


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "TrafficGenerator {:s}".format(self.id_object)
        return msg


