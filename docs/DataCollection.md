![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)


# Data collection and new functionalities

The results of a PyWiSim simulation are essentially raw data, with some minumum elaboration such as totals, mean and variance of some values, for primary conclusions. The main purpose of the PyWiSim simulator is to provide sets of raw data to be treated as required by the purpose of the simulation. This page describes the data collected by a PyWiSim simulation, the different entities where these data are collected, and how the collected data may be accessed in the objects comprising the simulation scenery.


## The simulation scenery

A proper knowledge of the simulation scenery structure, the entities involved and their relations, helps in the task of accessing data collected by the simulator. A simulation scenery involves the following entities:

- `BaseStation`, a radio base station.
    - `InterSliceSched`, assigns base station's resources to slices.
    - `Slice`, a set of user equipments and resources.
        - `Scheduler`, distributes resources among user groups.
        - `UserGroup`, several user equipments with a common profile.
            - `UserEquipment`, a user's data terminal equipment (DTE).
                - `Channel`, associated with each user user equipment.
                - `TransportBlock`, associated with each user equipment.
                - `PacketQueue`, the data packet queue for a user equipment.
                - `TrafficGenerator`, feeds the user equipment data queue.
        - `Resource`, communications resources assigned to this slice by the inter slice scheduler.

Hence, 

- a base station contains an interslice scheduler and several slices;
- a slice contains a scheduler, several user groups and a set of resources.
- a user group contains several user equipments.
- a user equipment contains a channel, a transport block, and a packet queue; a traffic generator is associated with the user equipment packet queue which it feeds.


## The simulation process

To run the simulation, the entities formerly described are organized in two lists, which serve as parameters to the Simulation object which effectively runs the simulation:

- `ls_slices` : a list of Slice objects.
- `ls_trfgen` : a list of TrafficGenerator objects.

These two categories of object, Slice and TrafficGenerator, generate events into the priority queue which controls the simulation. These lists are attributes of the Setup class.

On an execution instance, these actions may happen:

- each traffic generator feeds some packets into the user equipment data packet queue to which it is associated;
- the scheduler uses its own criteria to distribute its resources into its user equipments;
- with the resources assigned to it, the transport block assigned to each user equipment defines how to build one or more transport blocks, giving prioritry to packets in the user equipment retransmission queue, packets which were lost in previous transmissions;
- transport block transmission is simulated, which may result in success or failure;
- the user equipment is informed of transmission result; if successful packets included in the transport block transmitted are marked as sent and optionally preserved; if the transport block was lost, its packets are moved to a retransmission queue.

The first action is executed by the `run()` function in each traffic generator; the rest of the former actions are executed by the `run()` function in each slice.


## Access to data

The `ls_trfgen` list of traffic generators only purpose is to to generate packets into the data packet queues of the user equipments and schedule its next execution event in the simulator's piority queue; hence, it does not offer access to simulation data.

The most general way to access data is the `ls_slices` list, which comprises the slices in a base node. Attributes in the Slice object include:

- `ls_usrgrps` : a list of user groups in this slice.
- `ls_usreqs` : a list of user equipments in this slice.  
- `ls_res` : a list of resources available in this slice to assign to user equipments; this list is updated according to the inter slice scheduler actions.
- `ls_usreq_res` : a list of `[user equipment, resource, ... ]`. This list is updated according to the slice scheduler actions.

The list of user equipments allows access to each user equipment, and consquently to its channel, transport block, and packet queue.

To ease access and to follow the simulation dynamically, several "hook methods", i.e. methods intended to be rewritten, are called in the course of the simulation.

Slice method `run_ini(time_t)` is executed before Slice run actions, and Slice method `run_fin(time_t)` is executed after Slice run actions; both methods are invoked within the Slice `run(time_t)` function, one at the beginning and the other at the end, before the run function returns. The rewrite of these methods allows to capture data before and after a transmission instance. 

Traffic data by user equipment and by instant time `time_t` can be captured by rewritting the Slice method `transmit_tbs_fin(usreq, time_t, res_usreq, tbs_usreq)` which runs at the end of method `transmit_tbs` in class Slice. 

These methods are intended to be rewritten as needed to capture states before and after Slice run execution, which allows a close following of the simulation states. An example of use of these methods is provided in the example scheduler model, in the `models/scheduler/schedexample` directory. This example scheduler is intended to be a template for the development of a new scheduling algorithm, by copying the directory contents and modifiyng as needed. 


## Traffic data

Since the data packet queue handles data packets received and sent, it allows for the calculation of traffic performance indicators. A PacketQueue object containst two lists, one for received packets and another for successfully sent packets. Both lists contain the following items:

    [ id_packet, time received, time sent, packet ]

A packet may be an object, a string or a number of bits; PacketQueue method `size_bits(data_pkt)` returns the size in bits for a packet item, were it an object, a string or an integer; if an object is used, a `__len__(packet object)` function must be available.

When a packet is received, a list item as shown above is added to the received list, inserting a packet identifier, the time received, and the data packet itself (a number of bits, a string or an object); time sent is set to 0. When a packet is successfully transmitted, its time sent is set, and the list item is moved to the sent packets list. The PacketQueue class contains an attribute `keep_pkts` which determines if transmitted packets will be preserved or discarded, or how many will be kept (setting a maximum length of the sent list). Packets reported as lost in transmission are kept in a retransmission structure to give them priority in the next transmit instance.

As shown in the [Access to data](#Access_to_data) section, methods `run_ini()` and `run_fin()` in Slice can access the list of user equipments in the slice, `ls_usreqs`, and through it the PacketQueue objects of each user equipment, both before and after the slice `run()` actions; this allows to inspect the list of received and sent packets at each time of execution, as well as some traffic counters, both before and after `run()` actions. In the same way, method `transmit_tbs_fin()` allows access to traffic for each user equipment on each simulation instant time `time_t`.

Besides this very detailed access to each user equipment's traffic, the PacketQueue class keeps several counters in its `dc_traf` attribute, a dictionary of traffic counters which contains:

- number of received packets and number of received bits.
- number of sent packets and number of sent bits.
- number of lost packets and number of lost bits; this counter is incremented each time a packet is lost; if the same packet is lost several times, this counter is incremented on each loss instance.
- number of dropped packets and number of dropped bits, because maximum length of queue was exceeded.
- number of transport blocks created and number of bits included in them all.

Since the simulation time is always available, the throughput of each packet queue can be determined at any instant. 

The PacketQueue class also offers the possibility of keeping track of these traffic counters for the last k time instants, i.e. k units of time before the present time. This saves storage and allows to evaluate the transmission efficiency at the time needed.

## Adding new algorithms or functionalites

The rewrite of methods `run_ini()`, `run_fin()`, and `transmit_tbs_fin()` happens in a subclass of the Slice class, as methods of this subclass; this allows access to all attributes and  methods of the Slice upper class, and also to any additional attributes defined in the Slice subclass constructor, as well as to any additional methods defined in the Slice subclass. (Please note that a subclass of the original Slice class in the `libsimnet.basenode` module may be defined with the same name, Slice, in a different module, from which the new Slice class will be imported).

Hence, to add new functionality to the simulator, a developer may add in the Slice subclass whatever attributes and methods may be needed by the `run_ini()`, `run_fin()`, and `transmit_tbs_fin()` methods rewritten in the subclass. 

As an example, consider the writing of a new scheduling algorithm to distribute communications resources among user equipments. A new Scheduler class will be written, usually but not mandatory as a subclass of the `libsimnet.basenode` Scheduler class. In parallel, a Slice subclass inheriting from the `libsimnet.basenode` Slice class will be written, eventually adding new attributes and methods, plus the coding of the `run_ini()` and/or `run_fin()` and/or `transmit_tbs()` methods, which rewrite the corresponding methods in the parent Slice class. These methods have access to all attributes and methods of the Slice subclass and its superclass, as well as to objects linked to the Slice object. One of this linked objects will be the new Scheduler, which methods may be invoked from the Slice subclass, passing to them whatever data may be needed, and receiving from them the results of the new scheduler algorithm calculations. 

To follow the assignment of resources and transport blocks built for each user equipment and for all simulation times, the rewrite of the Slice method `transmit_tbs_fin()` provides the means, as shown in the simple example scheduler provided in directory `models/scheduler/schedexample`. A copy of this directory may serve as a startpoint for a new scheduler algorithm implementation.

For more detailed information please see the code documentation in start page  `html/index.html` of this repository. 


[Back to README](../README.md)

