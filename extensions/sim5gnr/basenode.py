#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# basenode.py: PyWiSim radio base station related entities
#

'''
PyWiSim, radio base station related entities module.
'''


import sys

from libsimnet.basenode import Slice,Resource,BaseStation
from extensions.sim5gnr.tables import loadSINR_MCStable,load_numerology_table,load_bands



class Slice(Slice):
    '''Represents a slice in a base station, a set of users and resources.
    '''
    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, id_slc, trans_delay, priority,numerology=2):
        '''Constructor.

        @param id_slc: an identifier for this slice.
        @param trans_delay: delay between transmissions.
        @param priority: order of execution.
        @param debug: if True prints messages.
        @param numerology: The 5G NR numerology used. Defines the TTI of the Slice. See tables/load_numerology_table.
        '''
        super().__init__(id_slc, trans_delay, priority)
        
        self.dc_usreq_q_delay = {}
        self.dc_usreq_traf = {}
        self.dc_usreq_channel = {}
        self.numerology = numerology
        num_table = load_numerology_table()
        self.trans_delay = num_table[self.numerology]["slot_duration"]
   
    def run_ini(self, time_t):
        for usreq in self.ls_usreqs:
            delay = 0
            count = 0
            bits = 0
            for pkt in usreq.pktque_dl.ls_recvd: 
                delay = delay +time_t-pkt[1]
                bits = bits+pkt[3]
                count += 1
            if count > 0 :
                avg_delay = delay/count
            else:
                avg_delay = 0
            if usreq.id_object in self.dc_usreq_traf.keys():
                self.dc_usreq_q_delay[usreq.id_object] += [ [time_t, bits, count,avg_delay] ]
            else:
                self.dc_usreq_q_delay[usreq.id_object] = [ [time_t, bits, count,avg_delay] ]
        return

 
    def transmit_tbs_fin(self, usreq, time_t, res_usreq, tbs_usreq, ul_dl):
        '''Rewrite to capture slice transmit_tbs actions.

        Allows to capture state before slice transmit_tbs actions.
        @param usreq: UserEquipment object.
        @param time_t: simulation instant time.
        @param res_usreq: number of resources for this user equipment.
        @param tbs_usreq: bits included in transport blocks for this user equipment.
        '''

        if usreq.id_object in self.dc_usreq_traf.keys():
            self.dc_usreq_traf[usreq.id_object] += [ [time_t, res_usreq, tbs_usreq] ]
            self.dc_usreq_channel[usreq.id_object] += [ [time_t, usreq.chan_state] ]
        else:
            self.dc_usreq_traf[usreq.id_object] = [ [time_t, res_usreq, tbs_usreq] ]
            self.dc_usreq_channel[usreq.id_object] = [ [time_t, usreq.chan_state] ]
        return

    def update_numelogy_based_on_delay(self,band,reqdelay):
        """This method sets Slice numerology depending on delay requirements."""
        bands = load_bands()
        fr = bands[band][0]
        if (fr == 2):  # FR2
            if reqdelay <= 2.5:
                self.trans_delay = self.num_table[3]
            else:
                self.trans_delay = self.num_table[2]
        else:  # FR1
            if reqdelay <= 5:
                self.trans_delay = self.num_table[2]
            elif reqdelay <= 10:
                self.trans_delay = self.num_table[1]
            else:
                self.trans_delay = self.num_table[0]


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "Slice 5g {:s}, {:s}:  delay {}, groups {:d}, res {:d}"
        msg = msg.format(self.id_slc, self.id_object, \
                self.trans_delay, len(self.ls_usrgrps), \
                len(self.ls_res) )
        return msg



class Resource(Resource):
    '''A communications resource.

    Determines the number of symbols that can be transmitted by this resource. 
    In 5G NR a Resource is a PRB. It is composed of 12 sub-bands, 1 slot and the 
    number of symbols per slot is by default 14. In case a long cycle prefix is used. In this case the symbols per slot is 12.
    In TDD if the PRB is used for downlink and uplink then syms_slot are used for DL and 14-syms_slot are used for UL.

    '''
    
    #def __init__(self, res_type="NoType", syms_slot=14, \
    #             nr_slots=1, nr_sbands=12,band="n257",dl=True,ul=False,long_cp=False):
    def __init__(self, res_type="NoType", res_pars=[]):
        '''Constructor.
        In 5g the resources can be used for UL, DL or both (in TDD bands). 
        In TDD a RB can be used for UL and DL. In this situation the number of symbols used for DL and UL is flexible and depends on
        the slot format configuration 38.213 v15.7 -Table 11.1.1-1: Slot formats for normal cyclic prefix
        In the simulator in this situation the user must implement two schedulers for the slice one for DL and the other for UL.
        The RB will be shared in the simulator for both schedulers. In this case the user in syms_slot must configure the symbols used for DL (syms_slot < 14).
        and when the UL scheduler ask for get_symbols() 14 - syms_slot will be used .

        @param res_type: the type of resource.
        @param syms_slot: number of symbols in a slot.
        @param nr_slots: number of slots.
        @param nr_sbands: number of sub bands. See tables/load_bands.
        @param band: The name of the 5G band. See tables/load_bands.
        @param dl: if the resource will be used for downlink or not.
        @param ul: if the resource will be used for uplink or not.
        @param long_cp: If long cyclic prefix will be used or not. If it is true the slot contains 12 symbols rather than 14.
        '''
        self.res_type= res_type
        self.syms_slot = 14
        self.nr_slots = 1
        self.nr_sbands = 12
        self.band = "n257"
        self.dl = True  #The resource can be used for downlink, uplink  or both
        self.ul = False  # The resource can be used for downlink, uplink  or both
        self.long_cp = False # In numerology 2 the ciclic prefix can ba a long CP
        if res_pars:
            self.syms_slot, self.nr_slots, self.nr_sbands, self.band, \
                self.dl, self.ul, self.long_cp = res_pars

        super().__init__(res_type, [self.syms_slot, self.nr_slots, self.nr_sbands])
        self.res_type = res_type  #Ver resource type en UserGroup #####
        self.freq_id = self.band # name of the band in 5g Nr
        self.bands = load_bands()
        try:
            self.tdd = self.bands[self.band][2]    # If the system is using TDD (True) or FDD(False)
            self.fr = self.bands[self.band][0]    # frequency range designation 1 (410 MHz – 7125 MHz) or 2(>24 Ghz)  
        except KeyError:
            print(" Bands ", self.bands)
            print("Band ",self.band," not in table. Please enter the band in the bands table.")
            sys.exit(0)
        
        self.nr_slots = 1       # number of time slots in each resource. In 5G the resources is asociated with one time slot.
                                # Dependeing on the numerology the time slot has different durations.
        self.nr_sbands = 12      # number of sub-bands per resource block. In 5G is set to 12.
        self.set_initial_config(self.syms_slot)      # symbols by time slot
        return
    
    
    def set_initial_config(self,syms_slot):
        """This method sets initial Resource configuration according to user configuration."""

        if self.tdd == False:
            if self.long_cp:
                self.syms_slot = 12
            else:
                self.syms_slot = 14
        else:
            if (self.dl and not self.ul) or ( not self.dl and self.ul): # if the resource will be used only for dl or only for ul   
                if self.long_cp:
                    self.syms_slot = 12
                else:
                    self.syms_slot = 14
            else:
                self.syms_slot = syms_slot
                #only in TDD and when the resource is used for dl and ul the user must specify syms_slot, the symbols used for DL. The symbols used for UL are 14 -syms_slot.
        return
        
    def get_symbols(self,ul_dl="DL"):
        '''Determines quantity of symbols accepted for transmission.
        
        @param ul_dl: if the resource is used for DL or UL scheduler. If this resource is shared by DL and UL schedulers, 
        and now will be used for schedule UL resources, the syms_slot will be set to 14 -self.syms_slot
        @return: quantity of symbols accepted for transmission.
        '''
        if ul_dl == "UL" and (self.dl==True and self.ul ==True):
            nr_syms = (14-self.syms_slot) * self.nr_slots * self.nr_sbands    
        else:
            nr_syms = self.syms_slot * self.nr_slots * self.nr_sbands
        
        return nr_syms

    
    #Esta funcion hay que ver como usarla de acuerdo a como se manejen los recursos compartidos DL/UL
    def update_config_QoS_based(self,reqDLconnections,reqULconnections,reqThroughputDL,reqThroughputUL):
        """This method sets initial Slice configuration according to service requirements."""
        if ( self.tdd):
            if reqDLconnections > 0 and reqULconnections == 0:
                self.syms_slot = 14
            if reqULconnections > 0 and reqDLconnections == 0:
                self.syms_slot = 14
            if reqULconnections > 0 and reqDLconnections > 0:
                DLfactor = float(reqDLconnections * reqThroughputDL) / (
                    reqDLconnections * reqThroughputDL
                    + reqULconnections * reqThroughputUL
                )
                self.syms_slot = int(14 * DLfactor)
                print(DLfactor,self.syms_slot)


    def __str__(self):
        '''For pretty printing.
        '''
        msg = "Resource 5g {:s}, type {:s}, syms {:d}, slots {:d}, sbands {:d},band {:s}, dl {:d}, ul {:d}, long_cp {:d}". \
                format(self.id_object, self.res_type, self.syms_slot, \
                    self.nr_slots, self.nr_sbands,self.freq_id,self.dl,self.ul,self.long_cp) 
        return msg



