![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)


# Simulation scenery

The simulation scenery includes all the objects and their relations required to model a wireless communications network; includes equipment, protocolos, and geography. This page gives some hints on how to simplify the build up of a simulation scenery.


## How the simulation works

The simulation is run by function `Simulation.simnet_run()` in module `libsimnet.simulator`. To create a `Simulation` object, the simulation scenery must be made available in either one of these ways:

- a `Setup` object, from the `libsimnet.simulation` module, or
- two lists, one with all + TrafficGenerator` objects, and the other with all the `Slice` objects. Each `Slice` object in the `Slice` objects list contains all necessary relations to the other objects required for the simulation: scheduler, user groups, their user equipments with their channel, etc. 

The simulation is run by method `Simulation.simnet_run()` in module `libsimnet.simulator`. Method `simnet_run()` may receive as a named parameter either a `Setup` object, or the two lists.

The following sections provide a brief explanation on each of these ways to define a simulation scenery. For more information, please see the code documentation in the `libsimnet.simulator` module.


## Scenery definition with lists

Script `extensions/simplesim/mk_simsetup.py` shows how to setup a very simple simulation scenery. This script creates only one object per class, and establishes the necessary relations among those objects. The objects created are collected in the two lists mentioned above:

- `ls_trfgen`, which includes all `TrafficGenerator` objects, and
- `ls_slices`, which includes all `Slice` objects, where each Slice object contains links to its associated objects. 

The former two lists contain all the objects and relations which define the simulation scenery, on which the simulation will be run.  These two lists are passed as parameters to method `simnet_run()` in a `Simulation` object to effectively run the simulation.

The `mk_simsetup.py` script creates just one object of each class; it is provided as an example or a sort of template to ease the task of creating objects and their relation. It shows the objects required, their parameters, and the relations that must be established among them to define a complete simulation structure.

The code in the `mk_simsetup.py` script is not optimized for execution, but for showing how to manually build a simulation scenery. To this end, some redundancy may be detected, e.g. by getting pointers to objects which in this context are already available but may be necessary in the general case.

For more information please see the `mk_simsetup.py` code documentation.


## Scenery definition with a Setup object

The creation of a simulation scenery may be simplified by the use of a help class called `Setup` in module `libsimnet.simulator`. This class creates all objects needed and their relations by reading entries in a list, where each item contains a class name, the number of items to create of this class, an object identifier to which these new objects must be assigned, and some parameters to define the scenery. The `Setup` class contains a method `change_attrs()` which eases configuration, allowing to change any attribute value in any of the objects which comprise the scenery.

Scripts `qa_simulator.py` and `mk_simplescene` in the `extensions/simplesim` module provide an example on how to setup and run a simple simulation using the class `Setup` to build up the scenery, and the class `Simulation` to effectively run the simulation.

The functions in `qa_simulator.py` provide an easy way to test simulations in an example case comprising:

- 1 base station;
- 2 slices;
- 3 user groups, 2 assigned to slice 1, 1 assigned to slice 2.

The number of user equipments, resources and other entities, as well as parameters for their creation can be set at will; most of these parameters have default values to simplify testing.

For more information on the `Setup` class please see the `libsimnet.simulator.Setup` code documentation. To see how the `Setup` class is used in a simple simulation example, please see the code documentation in script `extensions.simplesim.qa_simulator.py` and `extensions.simplesim.mk_simplescene.py`. These two scripts may be copied and customized to define a new simulation scenery quite easily.


[Back to README](../README.md)



