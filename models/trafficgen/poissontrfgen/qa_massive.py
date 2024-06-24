#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# qa_massive : tests a round robin scheduler example

'''Tests a round robin scheduler.
'''

# import classes from the main library
from libsimnet.simulator import mk_imports
from libsimnet.libutils import run_qa_tests

# import concrete classes overwritten from the main library abstract classes
from models.channel.randfixchan.randfix_chan import Channel, ChannelEnvironment
from poisson_trfgen import TrafficGenerator
from models.transpblock.simpletrblk.simple_trblk import TransportBlock 
from models.scheduler.rrscheduler.rr_scheduler import Scheduler      # the scheduler under test

# import functions for a simple simulation scenery
from extensions.simplesim.mk_simplescene import setup_sim, run_sim


def mk_scenery(nr_usreqs, nr_res, assgn_mat, dc_trfgen):
    '''Makes Setup object, adjusts attributes of some objects.

    @param dc_trfgen: attributes to change in TrafficGenerator objects.
    '''
    # make Setup object and entities first to change attributes
    setup_obj, ls_setup = setup_sim(nr_usreqs, nr_res, assgn_mat)
    ls_slices, ls_trfgen, dc_objs = setup_obj.mk_entities(debug=True)
    # add dc_profile to UserGroup, change Channel, TrafficGenerator
    dc_ugr_prof = {"lim_res":2, "res_type":"Good"}
    dc_chan_pars = {"chan_mode":"Fixed", "val_1":100}
    #dc_trfgen = {"debug":True, "size_dist":"Exponential"}
    for ugr in ["UG-1", "UG-2", "UG-3"]:
        setup_obj.change_attrs("UserGroup", ugr, {"dc_profile":dc_ugr_prof} )
        setup_obj.change_attrs("Channel", ugr, dc_chan_pars)
        setup_obj.change_attrs("TrafficGenerator", ugr, dc_trfgen)
    # change Scheduler debug attribute to print messages
    dc_sched_pars = {"debug":False}
    for slc in ["SL-1", "SL-2"]:
        setup_obj.change_attrs("Scheduler", slc, dc_sched_pars)
    return setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs


def test_mass_vrfy(dc_trfgen):
    '''Setup and run a long simulation with many users, verify.

    This test creates the Setup object and makes its entitities to change attributes not included in the parameters list of the run_sim() function.
    '''
    time_sim = 10                # simulation duration
    nr_usreqs = [1, 0, 0]       # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [500, 500, 500]  # packet size 
    gen_delay = [1, 1, 1]       # generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [300, 300, 300]   # queue max length per user group
    ug_keep = [True, True, True]  # keep packets after transmission
    ug_pkts = [1, 1, 1]        # packets to generate on each round
    nr_res = [0, 0, 4]          # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 4]]    # Good is 11*8*1=88, other: 14*1*12=168
    assgn_mat = []
    dbg_setup = [False, False, True]  # debug on slices, user eqs, traffic gens
    dbg_run = False
    setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs = \
        mk_scenery(nr_usreqs, nr_res, assgn_mat, dc_trfgen)
    # run simulation
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run, 
        ls_scene=[setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs])
    return


def test_mass_run(dc_trfgen):
    '''Setup and run a long simulation with many users, run.

    This test creates the Setup object and makes its entitities to change attributes not included in the parameters list of the run_sim() function.
    '''
    time_sim = 5              # simulation duration
    nr_usreqs = [1, 1, 1]       # user eqs qty per user group
    loss_prob = 0.0             # probability of losing packets
    gen_size = [500, 500, 500]  # packet size 
    gen_delay = [1, 1, 1]       # packet size and generation delay
    trans_delay = 1             # transmission steps
    ug_maxlen = [300, 300, 300]   # queue max length per user group
    ug_keep = [False, False, False]  # keep packets after transmission
    ug_pkts = [1, 1, 1]        # packets to generate on each round
    nr_res = [0, 0, 4]          # quantity of resources Poor, Fair, Good
    assgn_mat = [
        ["SL-1", "Good", 4]]    # Good is 11*8*1=88, other: 14*1*12=168
    assgn_mat = []
    dbg_setup = [False, False, False]  # setup debug
    dbg_run = True
    setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs = \
        mk_scenery(nr_usreqs, nr_res, assgn_mat, dc_trfgen)
    # run simulation
    run_sim(time_sim, nr_usreqs, loss_prob, 
        gen_size, gen_delay, ug_pkts,
        ug_maxlen, ug_keep,
        nr_res, assgn_mat,
        trans_delay, 
        dbg_setup, dbg_run, 
        ls_scene=[setup_obj, ls_setup, ls_slices, ls_trfgen, dc_objs])
    return


# docstring for pydoctor out of function
ls_tests = []
'''List of tests to show in menu.'''

if __name__ == "__main__":

    print("Poisson traffic generator, testing.")
    print("Creates a scenery of 1 BaseStation, 2 Slices and 3 UserGroups;")
    print("  number o UserEquipments and entities attributes are configurable.")
    # replace initially imported classes with overwritten classes
    mk_imports(nm_channel=Channel, nm_chanenv=ChannelEnvironment, \
        nm_trblk=TransportBlock, nm_trfgen=TrafficGenerator, \
        nm_scheduler=Scheduler)
    # execution parameters
    dc_trfgen_fx_db = {"debug":True, "size_dist":"Fixed"}
    dc_trfgen_ex_db = {"debug":True, "size_dist":"Exponential"}
    dc_trfgen_fx = {"debug":False, "size_dist":"Fixed"}
    dc_trfgen_ex = {"debug":False, "size_dist":"Exponential"}
    ls_tests = [ \
        ["Massive test fixed, verify.", test_mass_vrfy, dc_trfgen_fx_db], \
        ["Massive test exponencial, verify.", test_mass_vrfy, dc_trfgen_ex_db], \
        ["Massive test fixed, run.", test_mass_run, dc_trfgen_fx], \
        ["Massive test exponential, run.", test_mass_run, dc_trfgen_ex], \
        ] 
    print("Choose option, 'q' to quit:")
    op = run_qa_tests(ls_tests)



