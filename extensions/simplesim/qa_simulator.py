#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# qa_simulator : tests the simple simulator extension

'''Simple simulator tests.

This script creates a simple simulation scenery, configs the entities attributes, and runs some tests. Beyond its testing purposes, this script and related module mk_simsetup.py provide a template to customize for different simulation sceneries: just copy these two files and change as desired. 
'''

# import classes from the main library
from libsimnet.simulator import Setup, Simulation, mk_imports
from libsimnet.results import Statistics
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.channel.randfixchan.randfix_chan import Channel, ChannelEnvironment
from models.transpblock.simpletrblk.simple_trblk import TransportBlock
from models.trafficgen.simpletrfgen.simple_trfgen import TrafficGenerator
from models.scheduler.simplesched.simple_sched import Scheduler

# import functions for a simple simulation scenery
from mk_simplescene import setup_sim, run_sim

# Python imports
import sys
import numpy as np


def test_show():
    '''Create a simple simulation scenery and show its entities.
    '''
    setup_obj, ls_setup = setup_sim()
    ls_slices, trfgen, dc_objs = setup_obj.mk_entities(debug=False)

    ### verification of simulation scenary
    setup_obj.show_slices("BsStat-0", debug=3)
    setup_obj.show_trfgens()
    setup_obj.show_dc_objs()
    return


def test_change_attrs():
    '''Changes entities attributes on the created scenery.
    '''
    setup_obj, ls_setup = setup_sim()
    ls_slices, trfgen, dc_objs = setup_obj.mk_entities(debug=False)

    # change Slice attributes
    setup_obj.change_attrs("Slice", "SL-1", {"trans_delay":2, "debug":True})

    # change UserGroup attributes
    setup_obj.change_attrs("UserGroup", "UG-1", {"dc_profile":{"prof_nr":1}})

    # change UserEquipment attributes for one UserGroup
    dc_pars = {"v_pos":[3,3,3], "v_vel":[3,3,3]}
    setup_obj.change_attrs("UserEquipment", "UG-3", dc_pars)

    # change PacketQueue attributes for all UserEquipments in all Slices
    dc_pars = {"keep_pkts":False}
    setup_obj.change_attrs("PacketQueue", "SL-1", dc_pars)
    setup_obj.change_attrs("PacketQueue", "SL-2", dc_pars)

    # change Channel attributes for all UserEquipments in one Slice
    dc_pars = {"loss_prob":0.3}
    setup_obj.change_attrs("Channel", "SL-1", dc_pars)

    # change TransportBlock attributes for all UserEquipments in one UserGroup
    dc_pars = {"min_size":40, "max_size":300}
    setup_obj.change_attrs("TransportBlock", "UG-3", dc_pars)

    # change TrafficGenerator attributes for all UserEquipments in one UserGroup
    dc_pars = {"gen_size":40, "gen_delay":2}
    setup_obj.change_attrs("TrafficGenerator", "UG-3", dc_pars)

    setup_obj.show_slices("BsStat-0")
    return


def test_move():
    '''Runs a simulation with movement of user equipments.
    '''
    time_sim = 10
    # create scenery
    setup_obj, ls_setup = setup_sim()
    ls_slices, trfgen, dc_objs = setup_obj.mk_entities(debug=False)
    # initialize position and velocity vectors in user equipments
    setup_obj.change_attrs("UserEquipment", "UG-1", \
        {"v_pos":np.array([0,0,0]), "v_vel":np.array([1,1,0])})
    setup_obj.change_attrs("UserEquipment", "UG-2", \
        {"v_pos":np.array([0,0,0]), "v_vel":np.array([2,2,0])})
    setup_obj.change_attrs("UserEquipment", "UG-3", \
        {"v_pos":np.array([0,0,0]), "v_vel":np.array([3,3,0])})
    setup_obj.show_slices("BsStat-0")
    # run simulation directly using the libsimnet.simulator.Simulation class
    sim_obj = Simulation(time_sim, setup_obj=setup_obj)
    sim_obj.simnet_run(debug=False)
    # gather results and show
    for slc in ls_slices:
        print(slc)
        for usreq in slc.ls_usreqs:
            print("    {}".format(usreq))
    return


def test_run():
    '''Runs a simulation on the simple simulation scenery.
    '''
    time_sim = 10
    # create scenery
    setup_obj, ls_setup = setup_sim()
    ls_slices, trfgen, dc_objs = setup_obj.mk_entities(debug=False)
    # config attributes of objects in scenery
    for ugr in ["UG-1", "UG-2", "UG-3"]:
        setup_obj.change_attrs("Channel", ugr, {"loss_prob":0.4})
        setup_obj.change_attrs("UserEquipment", ugr, {"debug":3})
        setup_obj.change_attrs("UserEquipment", ugr, {"make_tb":"OneTBallRes"})
    # run simulation directly using the libsimnet.simulator.Simulation class
    sim_obj = Simulation(time_sim, setup_obj=setup_obj)
    sim_obj.simnet_run(debug=True)
    # gather results and show
    stat_obj = Statistics(sim_obj)  # create Statistics object
    print("\n=== Simulation results, queues final state")
    stat_obj.show_sent_rec()
    print("\n=== Simulation results, total counters and queues throughput")
    stat_obj.sim_totals(show=True)
    stat_obj.show_usereq_stats()
    return


def test_assgn_res():
    '''Changes resource assignment on the created scenery.
    '''
    time_sim = 5
    # create scenery
    setup_obj, ls_setup = setup_sim()
    ls_slices, trfgen, dc_objs = setup_obj.mk_entities()
    # config attributes of objects in scenery
    for ugr in ["UG-1", "UG-2", "UG-3"]:
        setup_obj.change_attrs("Channel", ugr, {"loss_prob":0.4})
        setup_obj.change_attrs("UserEquipment", ugr, {"debug":True})
    # create simulation object an assign resources
    sim_obj = Simulation(time_sim, setup_obj)
    print("=== Initial resources assignment to slices")
    setup_obj.show_slices("BsStat-0")
    print("\n=== New resources assignment to slices")
    assgn_mat = [
        ["SL-1", "Poor", 3],
        ["SL-2", "Poor", 2],
        ["SL-1", "Fair", 2],
        ["SL-2", "Fair", 2],
        ]                               # new resource assign matrix
    bs = setup_obj.dc_objs["BsStat-0"]  # BaseStation object
    # unassign resources
    bs.inter_sl_sched.unassign_res(bs.dc_res, bs.dc_slc)
    # reassign resources
    bs.inter_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)
    # show new resource assignment
    setup_obj.show_slices("BsStat-0")
    return


def test_mass_vrfy():
    '''Test with many users and long time, verify with a short time.
    '''
    time_sim = 5                     # simulation duration
    nr_usreqs = [1, 1, 2]            # user eqs qty per user group 2,2,2
    loss_prob = 0.0                  # probability of losing packets
    gen_size = [40, 40, 40]          # packet size 
    gen_delay = [1, 1, 1]            # time units between packet generation
    trans_delay = 1                  # time units between transmissions
    ug_maxlen = [0, 0, 0]            # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [3, 3, 3]              # packets to generate on each time
    nr_res = [2, 2, 2]               # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Poor", 1], ["SL-1", "Fair", 1], ["SL-1", "Good", 0],
        ["SL-2", "Poor", 1], ["SL-2", "Fair", 1], ["SL-2", "Good", 0]
        ]
    dbg_setup = [True, True, True]       # setup debug
    dbg_run = True
    # change simulation parameters and run with help function in mk_simplescene
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run)
    return


def test_mass_run():
    '''Test with many users and long time, run.
    '''
    time_sim = 1000                  # simulation duration
    nr_usreqs = [50, 50, 100]        # user eqs qty per user group 2,2,2
    loss_prob = 0.0                  # probability of losing packets
    gen_size = [40, 40, 40]          # packet size 
    gen_delay = [1, 1, 1]            # time units between packet generation
    trans_delay = 1                  # time units between transmissions
    ug_maxlen = [0, 0, 0]            # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [4, 4, 4]              # packets to generate on each time
    nr_res = [60, 70, 70]            # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Poor", 30], ["SL-1", "Fair", 35], ["SL-1", "Good", 35],
        ["SL-2", "Poor", 30], ["SL-2", "Fair", 35], ["SL-2", "Good", 35]
        ]
    dbg_setup = [False, False, False]   # setup debug
    dbg_run = False
    # change simulation parameters and run with help function in mk_simplescene
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run)
    return


# docstring for pydoctor out of function
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    print("Simple simulation scenery and tests\n")
    print("Creates a scenery of 1 BaseStation, 2 Slices and 3 UserGroups;")
    print("  number o UserEquipments and entities attributes are configurable.")
    # replace initially imported classes with overwritten classes
    mk_imports(nm_channel=Channel, nm_trblk=TransportBlock,
        nm_chanenv=ChannelEnvironment, nm_trfgen=TrafficGenerator,
        nm_scheduler=Scheduler)

    ls_tests = [ 
        ("Setup simulation and show.", test_show),
        ("Setup and change entities attributes.", test_change_attrs),
        ("Setup and test user equipment movement.", test_move),
        ("Setup and run simulation.", test_run),
        ("Base station, change resource assignment.", test_assgn_res),
        ("Massive test, verify.", test_mass_vrfy),
        ("Massive test, run.", test_mass_run),
        ] 
    run_qa_tests(ls_tests)

