#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# libsimnet.mk_simsetup : example template to setup a simulation scenery

'''PyWiSim template to setup a simulation scenery.

Creates the lists of objects necessary to run the simulation; these lists are passed as parameters to a Simulation object, which runs the simulation. 

This example template creates only one object of each class; it is just an example of object creation, their required parameters, and how to attach each new object to previuosly created objects to define a complete simulation structure.

This code is not optimized for execution, but for showing how to manually build a simulation scenery. To this end, some redundancy may be detected, e.g. by getting pointers to objects which in this context are already available, but may be necessary in the general case.
'''

### imports to access classes which define the simulation scenery
# import classes from the main library
from libsimnet.simulator import Simulation
from libsimnet.results import Statistics
from libsimnet.pktqueue import PacketQueue
from libsimnet.basenode import BaseStation, InterSliceSched, Slice, UserGroup, \
    Resource, TIME_UNIT, TIME_UNIT_S
from libsimnet.usernode import UserEquipment

# import concrete classes overwritten from abstract classes
from statistical_fading_channel import Channel 
from statistical_fading_chenv import ChannelEnvironment
from models.transpblock.simpletrblk.simple_trblk import TransportBlock
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from models.scheduler.simplesched.simple_sched import Scheduler

# Python imports
from queue import PriorityQueue
from time import perf_counter
import sys

import numpy as np

def set_initial_pos_vel(ls_usreqs,x_min=-100,x_max=100,y_min=-100,y_max=100,vel_lim_x_min= 0,vel_lim_x_max=30,vel_lim_y_min= 0,vel_lim_y_max = 30):
    for ue in ls_usreqs:
        x_t = np.random.uniform(x_min, x_max)
        y_t = np.random.uniform(y_min, y_max)
        pos = [x_t,y_t,2]
        if np.abs(x_t) > x_min+(x_max-x_min)/2:
            vx_t = np.random.uniform(vel_lim_x_min, vel_lim_x_max)*-np.sign(x_t)
        else:
            vx_t = np.random.uniform(vel_lim_x_min, vel_lim_x_max)*np.sign(x_t)
        if np.abs(y_t) > y_min+(y_max-y_min)/2:
            vy_t = np.random.uniform(vel_lim_y_min, vel_lim_y_max)*-np.sign(y_t)
        else:
            vy_t = np.random.uniform(vel_lim_y_min, vel_lim_y_max)*np.sign(y_t)
        vel = [vx_t,vy_t,0]
        ue.v_pos = np.array(pos)
        ue.v_vel = np.array(vel)
        print("------pos, vel-----",ue.v_pos,ue.v_vel)
    


### set some simulation variables
time_sim = 1000
'''Number of times to run simulation (number of simulation intervals).'''
gen_delay = 1
'''Delay between packet generation.'''
trans_delay = 1
'''Delay between transmisions.'''


### collections of objects required to setup the simulation scenery
ls_trfgen = []
'''List of traffic generators.'''
ls_slices = []
'''List of Slices.'''
dc_objs = {}
'''Container entities by identificator.'''

debug = True    # to show the simulation scenery


### BaseStation
# create a BaseStation object
id_base = "BaseStationOne"     # BaseStation name
pt_bs = BaseStation(id_base)   # creates with unique identifier BS-1
dc_objs["BS-1"] = pt_bs        # add to entities dictionary


### ChannelEnvironment
# create a channel environment to which channels will be attached
pt_ch_env = ChannelEnvironment(fcGHz=30,number_prbs=100,bw_prb=180000,posx_min=-500,posx_max=500,posy_min=-500,posy_max=500,grid_number=50,bspos=[0,0,20],Ptx_db=30,sigma_shadow=5,shadow_corr_distance=5,order =4,ray_ric = "Rayleigh",number_sin=10,K_LOS=10)


### InterSliceScheduler
# create an InterSliceScheduler, unique for each BaseStation
priority = 4    # order of event execution in a certain instant time
debug = True    # debug for InterSliceScheduler
pt_inter_sl_sched = InterSliceSched(priority, debug)
# attach to BaseStation 1, BS-1
pt_bs = dc_objs["BS-1"]    # get pointer to BaseStation to attach to
pt_bs.inter_sl_sched = pt_inter_sl_sched   # attach to BaseStation


### Resource
# repeat for all Resources to create for a BaseStation
attach_to = "BS-1"      # BaseStation unique identifier
id_pref = "BS-1:Res-"   # prefix for a name; unique identifier RS-1
res_type = 'Fair'       # Resource type
syms_slot = 11          # number of symbols in a slot
nr_slots = 4            # number of slots
nr_sbands = 1           # number of sub bands
nr_items = 1            # number of resource objects to create
dc_objs[attach_to].mk_dc_res(res_type, nr_items,  
    [syms_slot, nr_slots, nr_sbands] )  # makes Resources, attach to BaseStation

if debug:
    print("=== Simulation scenery ===\n")
    print("--- Base station and associated entities")
    print(pt_bs)
    print("        ", pt_bs.inter_sl_sched)
    for key in pt_bs.dc_res:
        print("        ", pt_bs.dc_res[key][0][0])


### Slice
# create a Slice, attach to a BaseStation
# repeat for all Slices attached to the BaseStation
id_slc = "Slc-1:BS-1"   # Slice name; unique identifier "SL-1"
trans_delay = 1         # time between successive transmisions
priority = 2            # order of event execution in a certain instant time
pt_slc = Slice(id_slc, trans_delay, priority)    # pointer to Slice
dc_objs["SL-1"] = pt_slc    # add to entities dictionary
ls_slices += [pt_slc]       # list of all Slices
# attach Slice SL-1 to BaseStation BS-1
pt_bs = dc_objs["BS-1"]     # get pointer to BaseStation to attach to
pt_bs.dc_slc[pt_slc.id_object] = pt_slc     # attach to BaseStation

if debug:
    print("    ", pt_slc) #, pt_obj.id_slc)


### Scheduler
# create Schedulers DL y optionally UL, attach to Slice
# repeat for all Slices attached to the BaseStation
pt_sched_dl = Scheduler()      # create scheduler DL unique id SL-1
pt_slc.sched_dl = pt_sched_dl  # attach to Slice
#pt_sched_ul = Scheduler()      # create scheduler DL unique id SL-1
#pt_slc.sched_ul = pt_sched_ul  # attach to Slice

if debug:
    print("        ", pt_sched_dl)
    #print("        ", pt_sched_ul)


### UserGroup
# create a UserGroup, attach to a Slice
# repeat for all UserGroups in the Slice
id_usr_grp = "UG-1:SL-1:BS-1"   # UserGroup name; unique identifier UG-1
profile = "UG-profile"          # UserGroup profile
pt_ugr = UserGroup(id_usr_grp, profile)     # pointer to UserGroup
dc_objs["UG-1"] = pt_ugr        # add to entities dictionary
dc_objs["SL-1"].ls_usrgrps += [pt_ugr]  # add to list of UserGroup in Slice
                

### UserEquipment
# create a UserEquipment, attach to a UserGroup
# repeat for all UserEquipments in the UserGroup
pt_ugr = dc_objs["UG-1"]    # UserGroup to attach UserEquipment to
v_pos = [5, 5, 2]           # UserEquipment position vector
v_vel = [30, 30, 0]           # UserEquipment velocity vector
usreq = UserEquipment(pt_ugr, v_pos, v_vel, debug=debug)    # id UE-1
dc_objs["UG-1"].ls_usreqs += [usreq]    # add to list in UserGroup
dc_objs["SL-1"].ls_usreqs += [usreq]    # add to list in Slice


### Channel
# create Channel, assign ChannelEnvironment to Channel, Channel to UserEquipment
# repeat for each UserEquipment
loss_prob = 0.0             # channel probability loss
usreq.chan = Channel(pt_ch_env, loss_prob)  # also assigns ChannelEnvironment


### TransportBlock
# create transport block
# repeat for each UserEquipment
usreq.tr_blk = TransportBlock()     # creates and assigns TransportBlock


### PacketQueue
# create PacketQueue for UserEquipment, attach to UserEquipment
# repeat for UL traffic in this UserEQuipment
# repeat for each UserEquipment in all UserGroups
qu_prefix="PQ-"     # PacketQueue prefix for unique identifier
pkt_prefix="Pkt-"   # packet prefix for this queue unique packet identifier
max_len=0           # maximum queue lenght, then drop packets
keep_pkts=True      # keep packets after successfully transmitted
usreq.pktque_dl = PacketQueue(qu_prefix, pkt_prefix, max_len, keep_pkts)


### TrafficGenerator
# create TrafficGenerator, attach to UserEquipment data PacketQueue
# repeat for UL traffic in this UserEQuipment, attach to UL PacketQueue
# repeat for each PacketQueue in each UserEquipment
gen_size = 30       # packet length in bits
gen_delay = 1       # time between successive packet generation
nr_pkts = 1         # number or packets to generate on each  instant time
priority = 1        # order of event execution in a certain instant time
pt_tg_obj = TrafficGenerator(usreq.pktque_dl, gen_size, \
            gen_delay, nr_pkts, priority, debug)
# add traffic generator to list of traffic generators
usreq.trf_gen = pt_tg_obj
ls_trfgen += [pt_tg_obj] # list of all traffic generators


### assign Resources to Slices
# the assign matrix assigns all available Resources to Slices
#    [ Slice id, Resource type, quantity to assign ]
assgn_mat = [ [ "SL-1", "Fair", 1] ]    # Resource assignment matrix
attach_to = "BS-1"              # BaseStation for Resource assignment 
bs = dc_objs[attach_to]         # BaseStation object
bs.inter_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    # assigns Resources to Slices according to the assign matrix

###The first call update random in a region
set_initial_pos_vel(pt_ugr.ls_usreqs,x_min=150,x_max=200,y_min=-100,y_max=-150) # random
###This second call assigns to a fixed possition and velocity
set_initial_pos_vel(pt_ugr.ls_usreqs,x_min=200,x_max=200,y_min=-100,y_max=-100,vel_lim_x_min= 30,vel_lim_x_max=30,vel_lim_y_min= 30,vel_lim_y_max = 30)#deterministic

### show simulation scenery
if debug:
    print(" "*8, pt_ugr)
    for ueq in pt_ugr.ls_usreqs:
        print(" "*12, ueq)
        print(" "*16, ueq.chan)
        print(" "*20, ueq.chan.ch_env)
        print(" "*16, ueq.tr_blk)
        print(" "*16, ueq.pktque_dl)
        print(" "*16, ueq.trf_gen)
    print("--- Traffic generators list")
    for trf_gen in ls_trfgen:
        print(" "*4, trf_gen, "Queue ", trf_gen.pktque.id_object)
    print("--- Slices list")
    for slc in ls_slices:
        print("    ", slc)
    print("--- Dictionary of objects")
    for key in dc_objs:
        print("    ", key, dc_objs[key])


    ### run simulation
    print()
    run_sim = input("Run simulation (y/n):")
    if run_sim != "y":
        sys.exit()
    dbg_run = True
    #gen_delay, trans_delay = 2, 1
    #sim_obj = Simulation(time_sim, gen_delay, trans_delay, \
    sim_obj = Simulation(time_sim, ls_trfgen=ls_trfgen, ls_slices=ls_slices)
    start_time, end_time = sim_obj.simnet_run(debug=dbg_run)
    stat_obj = Statistics(sim_obj)   # create Statistics object
    if dbg_run:
        print("\n=== Simulation results, queues final state ===")
        stat_obj.show_sent_rec()
    #print("    Simulation duration {:.3f} s".format(end_time - start_time))
    stat_obj.sim_totals(show=True)


