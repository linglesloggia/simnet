#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simulator : setup, configure and run the simulation

'''PyWiSim simulator module, to setup, configure and run the simulation.

Class Setup allows for the setup of the simulation scenery. It may be customized to implement different simulation sceneries. Classes for the required simulation entries must be imported in each of the simulator implementations, so as to allow the use of this setup to run classes with different simulator implementations.

Class Simulation runs a simulation scenery specified by a Setup object.
'''

# import classes from the main library
from libsimnet.basenode import BaseStation, InterSliceSched, Slice, UserGroup,\
    Resource, Scheduler, TIME_UNIT, TIME_UNIT_S
from libsimnet.usernode import UserEquipment, Channel, ChannelEnvironment,\
    TransportBlock, TrafficGenerator
from libsimnet.pktqueue import PacketQueue

# Python imports
from time import perf_counter
from queue import PriorityQueue


def mk_imports(**dc_classes):
    '''Replaces libsimnet classes for overwritten classes.

    Allows overwritten classes to substitute libsimet classes, which allows to include implementations of other communications protocols or algorithms. Class variables declared global necessary for effective substitution.
    @param dc_classes: a dictionary of {name:class}, the name of the class and the class itself.
    '''
    global Channel, ChannelEnvironment, TransportBlock, TrafficGenerator,\
        Scheduler, Slice
    if "nm_channel" in dc_classes:
        Channel = dc_classes["nm_channel"]
    if "nm_chanenv" in dc_classes:
        ChannelEnvironment = dc_classes["nm_chanenv"]
    if "nm_trblk" in dc_classes:
        TransportBlock = dc_classes["nm_trblk"]
    if "nm_trfgen" in dc_classes:
        TrafficGenerator = dc_classes["nm_trfgen"]
    if "nm_scheduler" in dc_classes:
        Scheduler = dc_classes["nm_scheduler"]
    if "nm_slice" in dc_classes:
        Slice = dc_classes["nm_slice"]
    return



class Setup:
    '''Sets up the simulation scenery.

This is a help class to set up and configure a simulation scenery. This includes:
    - setup: creating all objects of the different classes and establishing the relationships among them;
    - configuration: optionally and selectively updating the attribute values of the objects comprising the simulation scenery.

The simulation scenery is setup by reading entries in a list of the following structure::

    [setup_list] = [ [setup_item], ... ]
    setup_item = [ class_name, nr_items, attach_to, id_object ]

Where:
    - class_name: the name of the class of the object to create.
    - nr_items: the number of objects to create.
    - attach_to: the object to which the new objects will be attached.
    - id_object: a user defined name for a single object, required for BaseStation, Slice, UserGroup; besides these names, all entities have a unique object identifier automatically assigned based on a user given or default value prefix.

In the following list, indentation indicates objects created together with the upper item. The setup of a scenery involves the following entities:
    - BaseStation.
        - InterSliceSched, assigns the base station's resources to slices.
        - Slice, the slices in a base station, attached to a BaseStation.
            - Scheduler, to distribute resources among user groups, created together with and attached to a slice.
            - UserGroup, attached to a slice.
                - UserEquipment, created together with and attached to a UserGroup.
                    - Channel, the channel associated with each user.
                    - TransportBlock, the transport block associated with each user.
                    - PacketQueue, the data packet queue for a user equipment.
                    - TrafficGenerator, the traffic generator feeding the user equipment data queue; traffic generators are returned in a list.
            - Resource, the resources attached to a slice.

The former scheme allows for users of the same profile to be attached to a user group (all users in a user group have the same profile). Resources with different characteristics may be assigned to the same slice; for this reason they are not created simultaneously with the slice.

The outcome of this class are two lists, which serve as parameters to the Simulation object:
    - C{ls_slices : } a list of Slice objects.
    - C{ls_trfgen : } a list of TrafficGenerator objects.

These two categories of object, Slice and TrafficGenerator, generate events into the priority queue which controls the simulation.

Container entities BaseStation, Slice and UserGroup are recorded in a dictionary; this is needed to attach other entities to them, e.g. UserEquipment to UserGroup.

Please note that the use of this class is not mandatory; the simulation scenery may be set up by hand to obtain the lists ls_slices and ls_trfgen which must be given as parameters to the simulation object function which runs the simulation.
'''

    def __init__(self, ls_setup):
        '''Constructor.

        @param ls_setup: a list of configuration items to set up the simulation scenery.
        '''
        self.ls_setup = ls_setup
        '''List of configuration items to set up the simulation scenery.'''

        self.dc_objs = {}
        '''Container entities by identificator.'''
        self.ls_trfgen = []
        '''All traffic generators.'''
        self.ls_slices = []
        '''All slices.'''

        self.pt_ch_env = ChannelEnvironment()
        '''One ChannelEnvironment object for all UserEquipment objects.'''


    def mk_entities(self, debug=False):
        '''Makes entities and sets relations for all items in the setup list.

        @param debug: if True prints messages.
        @return: a list of slices, a list of traffic generators, a dictionary of some of the objects created.
        '''
        for setup_item in self.ls_setup:
            if debug:
                print("--- Making objects in ", setup_item)
            self.mk_ent_group(setup_item, debug)
        return self.ls_slices, self.ls_trfgen, self.dc_objs


    def mk_ent_group(self, setup_item, debug=False):
        '''Makes entities and sets relations for a specific class of object.

        @param setup_item: an item in the setup list.
        @param debug: if True prints messages.
        '''

        if setup_item[0] == "BaseStation":
            class_nm, nr_items, attach_to, id_pref = setup_item
            for i in range(0, nr_items):
                # create base station
                id_base = id_pref + str(i)
                pt_obj = BaseStation(id_base)
                self.dc_objs[id_base] = pt_obj   # user given name
                self.dc_objs[pt_obj.id_object] = pt_obj   # auto identifier

                # create inter slice scheduler
                pt_inter_sl_sched = InterSliceSched()
                pt_obj.inter_sl_sched = pt_inter_sl_sched

                if debug:
                    print(pt_obj, pt_obj.id_base)
                    print(pt_obj.inter_sl_sched)

        elif setup_item[0] == "Slice":
            class_nm, nr_items, attach_to, id_pref =  setup_item
            for i in range(0, nr_items):
                # create slice
                id_slc = id_pref + str(i)
                pt_obj = Slice(id_slc)
                self.dc_objs[id_slc] = pt_obj
                self.dc_objs[pt_obj.id_object] = pt_obj
                self.ls_slices += [pt_obj]    # list of all slices

                # attach slice to base station
                bs = self.dc_objs[attach_to]
                bs.dc_slc[pt_obj.id_object] = pt_obj

                # create scheduler
                pt_sched = Scheduler()        # only 1 scheduler per slice
                pt_obj.sched_dl = pt_sched       # attach scheduler to slice

                if debug:
                    print(pt_obj, pt_obj.id_slc)
                    print(pt_sched)

            self.dc_objs[attach_to].ls_slc = self.ls_slices

        elif setup_item[0] == "UserGroup":
            class_nm, nr_items, attach_to, id_pref, nr_users =  setup_item
            for i in range(0, nr_items):
                # create user group
                id_usrgrp = id_pref + str(i)
                pt_obj = UserGroup(id_usrgrp) #, profile)
                self.dc_objs[id_usrgrp] = pt_obj
                self.dc_objs[pt_obj.id_object] = pt_obj
                self.dc_objs[attach_to].ls_usrgrps += [pt_obj]

                # create user equipments in this user group
                for i in range(0, nr_users):
                    # create user equipment
                    usreq = UserEquipment(pt_obj)
                    # add user equipment to list of user equimpments
                    pt_obj.ls_usreqs += [ usreq ]  # add to list of UserEq
                    # create channel, assign channel environment
                    usreq.chan = Channel(self.pt_ch_env)
                    # create transport block
                    usreq.tr_blk = TransportBlock()
                    # create user equipment queue
                    usreq.pktque_dl = PacketQueue()
                    # create traffic generator, attach to user data queue
                    pt_tg_obj = TrafficGenerator(usreq.pktque_dl)
                    usreq.trf_gen = pt_tg_obj
                    # add traffic generator to list of traffic generators
                    self.ls_trfgen += [pt_tg_obj]  # all traffic generators

                ls_usreqs = pt_obj.ls_usreqs  # all user eqs in the user group
                self.dc_objs[attach_to].ls_usreqs += ls_usreqs
                if debug:
                    print(pt_obj, pt_obj.id_usrgr)
                    for ueq in ls_usreqs:
                        print("  ", ueq)

        elif setup_item[0] == "Resource":   # creates resources
            class_nm, nr_items, attach_to, id_pref,\
                res_type, syms_slot, nr_slots, nr_sbands = setup_item
            self.dc_objs[attach_to].mk_dc_res(res_type, nr_items,\
                [syms_slot, nr_slots, nr_sbands] )

        elif setup_item[0] == "ResAssign":  # assigns resources to slices
            class_nm, nr_items, attach_to, assgn_mat = setup_item
            bs = self.dc_objs[attach_to]    # BaseStation object
            bs.inter_sl_sched.assign_slc_res(bs.dc_slc, bs.dc_res, assgn_mat)

        else:
            print("ERROR, setup_item:", setup_item)

        return


    def change_attrs(self, nm_class, id_object, dc_pars={}, new_attrs=False):
        '''Change value of attributes in objects of the simulation scenery.

        @param nm_class: name of a class.
        @param id_object: object identifier.
        @param dc_pars: a dictionary of {attribute:value} to change the values of object attributes.
        @param new_attrs: adds new attributes.
        '''

        pt_obj = None
        if id_object in self.dc_objs:
            pt_obj = self.dc_objs[id_object]    # pointer to object 
        else:
            print("ERROR in change_attrs: id_object {} not in dc_objs".\
                format(id_object))
            return
        if nm_class in ["BaseStation", "Slice", "UserGroup"]:
            self.assgn_attrs(pt_obj, dc_pars)
            return
        if nm_class == "InterSliceScheduler" and type(pt_obj) is BaseStation:
            self.assgn_attrs(pt_obj.inter_sl_sched, dc_pars)
            return
        if nm_class == "Scheduler" and type(pt_obj) is Slice:
            self.assgn_attrs(pt_obj.sched_dl, dc_pars)
            return
        if type(pt_obj) is Slice or type(pt_obj) is UserGroup:
            ls_usreqs = pt_obj.ls_usreqs
            if nm_class == "UserEquipment":
                for usreq in ls_usreqs:
                    self.assgn_attrs(usreq, dc_pars)
            elif nm_class == "Channel":
                for usreq in ls_usreqs:
                    self.assgn_attrs(usreq.chan, dc_pars)
            elif nm_class == "PacketQueue":
                for usreq in ls_usreqs:
                    self.assgn_attrs(usreq.pktque_dl, dc_pars)
            elif nm_class == "TransportBlock":
                for usreq in ls_usreqs:
                    self.assgn_attrs(usreq.tr_blk, dc_pars)
            elif nm_class == "TrafficGenerator":
                for usreq in ls_usreqs:
                    self.assgn_attrs(usreq.trf_gen, dc_pars)
            else:
                print("ERROR change_attrs: nm_class {}, id_object {} not valid".\
                format(nm_class, id_object))
            return
        else:
            print("ERROR change_attrs: nm_class {}, id_object {} not valid".\
                format(nm_class, id_object))
            return


    def assgn_attrs(self, pt_obj, dc_pars, new_attrs=False):
        '''Assigns values to attributes of an object.

        @param pt_obj: a pointer to an object.
        @param dc_pars: a dictionary of {attribute:value} to change the values of object attributes.
        @param new_attrs: adds new attributes.
        '''
        for key in dc_pars:
            if key in pt_obj.__dict__:  # key is an attribute of this object
                pt_obj.__dict__[key] = dc_pars[key]
            elif new_attrs:
                pt_obj.__dict__[key] = dc_pars[key]
            else:
                print("ERROR: assgn_attrs, attribute not in class")
        return


    def show_dc_objs(self):
        '''Show container entities created.

        Container entities like BaseStation, Slice, UserGroup, are recorded by object identificator in a dictionary.
        '''
        print("\n--- Container entities dictionary:")
        for key in self.dc_objs.keys():
            print("  {:s}: {}".format(key, self.dc_objs[key]) )
        print()
        return


    def show_slices(self, bs_stat, debug=True):
        '''Show the list of all slices created in a base station.

        @param bs_stat: the identificator of a BaseStation.
        @param debug: if 3 prints detailed messages.
        '''
        # print("\nBaseStation: ", self.dc_objs[bs_stat] )
        # print("    InterSliceSched: ", self.dc_objs[bs_stat].inter_sl_sched )
        print("\n{}".format(self.dc_objs[bs_stat]) )
        print("    {}".format(self.dc_objs[bs_stat].inter_sl_sched) )
        for slc_key in self.dc_objs[bs_stat].dc_slc.keys():
            slc = self.dc_objs[bs_stat].dc_slc[slc_key]
            print("   ", slc)
            print("       ", slc.sched_dl)
            if slc.ls_res:
                print("        Resources: {}".format(len(slc.ls_res)))
                if debug == 3:
                    for res in slc.ls_res:
                        print("           ", res)
            for usrgrp in slc.ls_usrgrps:
                print("       ", usrgrp)
                for usreq in usrgrp.ls_usreqs:
                    print("           ", usreq)
                    print("               ", usreq.chan)
                    print("               ", usreq.tr_blk)
                    print("               ", usreq.trf_gen)
                    print("               ", usreq.pktque_dl)
                    """
                    usreq.pktque_dl.show_counters()
                    #print(" "*16 + "Received:")
                    usreq.pktque_dl.show_ls_pkts("Received")
                    #print(" "*16 + "Pending:")
                    usreq.pktque_dl.show_pending()
                    #print(" "*16 + "Retransmit:")
                    usreq.pktque_dl.show_retrans()
                    #print(" "*16 + "Sent:")
                    usreq.pktque_dl.show_ls_pkts("Sent")
                    """
            #print("        --- User equipments in slice")
            #for usreq in slc.ls_usreqs:
            #    print("           {}, {}".format(usreq, usreq.usr_grp))
        return


    def show_trfgens(self):
        '''Shows all traffic generators and the packet queues they feed.
        '''
        print("\nTraffic generators and their packet queues:")
        for trfgen in self.ls_trfgen:
            print("    {}".format(trfgen))
            print("        {}".format(trfgen.pktque))
        return



class Simulation:
    '''Runs the simulation.  

    Starts and stops simulation.
    '''

    def __init__(self, time_sim, setup_obj=None, \
            ls_trfgen=[], ls_slices=[], dc_actions={}):
        '''Constructor.

        @param time_sim: simulation duration.
        @param setup_obj: a Setup object with all the simulation scenery.
        @param ls_trfgen: list of traffic generators.
        @param ls_slices: list of slices.
        @param dc_actions: dictionary of actions, {name:function}.
        '''
        
        self.time_sim = time_sim
        '''Number of time units to run simulation.'''
        self.setup_obj = setup_obj
        '''Setup object on which to run the simulation.'''
        self.ls_trfgen = []
        '''List of traffic generators.'''
        self.ls_slices = []
        '''List of slices.'''
        self.dc_actions = dc_actions
        '''Dictionary of actions.'''

        self.start_time = 0.0
        '''Simulation run start time.'''
        self.end_time = 0.0
        '''Simulation run stop time.'''
        self.time_now = 0
        '''Simulation current execution instant.'''

        # initialize according to parameters received
        if setup_obj:
            self.ls_trfgen = setup_obj.ls_trfgen
            self.ls_slices = setup_obj.ls_slices
        elif ls_slices and ls_trfgen:
            self.ls_trfgen = ls_trfgen
            self.ls_slices = ls_slices
        else:
            print("Simulation. ERROR in constructor parameters received")
            return

        # initialize event driven simulator engine
        self.time_t = 1                     # current instant time
        '''Current instant time.'''
        self.event_qu = PriorityQueue()
        '''The event priority queue.'''

        # create and put in queue event to end simulation
        end_event = [self.time_sim+1, 1, "EndSimulation"]
        self.event_qu.put(end_event)

        # create list of tasks
        ls_tasks = []
        # create list of tasks for simulation progress
        ls_tasks += [ ["ShowProgress", 100, 0] ]
        self.dc_actions["ShowProgress"] = self.sim_progress
        # create list of tasks for traffic generators, priority 1
        for trfgen in self.ls_trfgen:
            ls_tasks += [ [trfgen.id_object, trfgen.gen_delay, 1] ]
            self.dc_actions[trfgen.id_object] = trfgen.run 
        # create list of tasks for slices in transmission, priority 2
        for slc in self.ls_slices:
            ls_tasks += [ [slc.id_object, slc.trans_delay, 2] ]
            self.dc_actions[slc.id_object] = slc.run
        # create list of tasks for interslice schedulers, priority 3
        #
        # execute initial actions, to generate a first event of each type
        for task in ls_tasks:
            id_action, delay, priority = task
            self.event_qu.put([delay, priority, id_action])

        return


    def sim_progress(self, time_t):
        '''Shows simulation progress.
        '''
        print("=== time {} {}".format(self.time_t, TIME_UNIT))
        next_event = [time_t+100, 0, "ShowProgress"]
        return next_event



    def simnet_run(self, debug=False):
        '''Runs events in the simulation.

        @param debug: if True prints messages.
        @return: simulation start and stop times.
        '''

        #if debug:
        print("\n=== Starting simulation ===")
        print("Time unit: {}; time unit in seconds: {}".\
            format(TIME_UNIT, TIME_UNIT_S) )
        print("Simulation duration {} {}\n".format(self.time_sim, TIME_UNIT))
        self.start_time = perf_counter()

        #  run event engine
        while not self.event_qu.empty():
            event = self.event_qu.get()
            next_time_t, priority, id_action = event
            if id_action == "EndSimulation":
                print("\n=== time {} {}, simulation ended.".\
                    format(self.time_t, TIME_UNIT)) 
                break
            else:
                if self.time_t == next_time_t:  # event executes now
                    pass
                else:
                    self.time_t = next_time_t   # event executes in future
            if self.time_now < self.time_t:    # new instance of execution
                if debug:
                    print("--- simulation time: {} {}".\
                        format(self.time_t, TIME_UNIT))
                self.time_now = self.time_t
            new_event = self.dc_actions[id_action](self.time_t)  # exec action
            if new_event:   # function returned an event
                self.event_qu.put(new_event)        # add new event to queue
            else:           # function returned nothing
                pass        # this function will no longer be executed

        self.end_time = perf_counter()
        if debug:
            duration = self.end_time - self.start_time
            print("Simulation duration: {:.3f}".format(duration) )        
            #print("=== Simulation finished ===\n")        
        return self.start_time, self.end_time



if __name__ == "__main__":
    print("To run tests please do:")
    print("    python3 extensions/simplesim/qa_simulator.py")
