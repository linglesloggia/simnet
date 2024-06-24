![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)


# The PyWiSim simulation engine

This document describes how PyWiSim uses Python queue.PriorityQueue to implement an event driven simulation engine.


## The Python priority queue

The Python class `queue.PriorityQueue` implements a queue in which the entries are kept sorted so that the lowest valued entry is retrieved first. Simulation happens in a discrete simulation time scale, which allows each process to insert its own action as an entry in the priority queue. The first item in the entry is an instant time in the simulation time scale, the instant time on which the process action is to be executed. In this way, different processes insert entries for their future actions with the certainty that the execution will happen in the correct order, i.e. earlier actions executed before later actions. Since all this happens in time, entries in the priority queue are here called "events".

Python priority queue class implements two main methods:
- `put(item)` : inserts an entry in the queue; since entries are kept ordered, this operation takes an O(log n) execution time, where n is the number of entries in the queue at the moment.
- `get()` : returns the event nearest in time to be executed. Since the queue is kept ordered, this operation takes O(1) excecution time.

The Python priority queue offers the most efficient mechanism possible for handling events happening in time.


## Events

In PyWiSim, an event indicates a future action to happen in an instant time t, and the action to be executed. An even is implemented as a tuple

 `[next_time_t, priority, id_action]`

where:
- `next_time_t` is the instant time for execution. This is determined by the function itself, which must return its next event when called.
- `priority` is a number indicating an order of execution for events with the same instant time; the lowest priority event is executed first.
- `id_action` is a string identifying a function or some action. Since entries in a priority queue must be compared to determine which is the nearest in time to be executed, a function cannot take this place.

Since functions cannnot be included in the event tuple, and its identifier goes instead, a  dictionary of function identifier to the executable function is used as a reference:
- `key` : an identifier of a function.
- `value` : the function itself.

When time comes to execute an event, i.e. present simulation time == next\_time\_t field in the event, the function is executed. The function receives the present instant time as a parameter, adds to it a delay according to its own logic, and returns a new event where the field `next_time_t` is the instant time for its future execution. The event returned by the function is inserted in the priority queue, in the corresponding order of execution guaranteed by the priority queue Python implementation.

As said, if there are several events to be executed in this instant time, the tuple structure ensures the one with lowest priority will be executed first.


## The simulation sequence

On initialization, the event priority queue receives an event `[time_sim+1, 1, "EndSimulation"]`, where `time_sim` is the total simulation duration, priority is 1, and the label "EndSimulation" indicates the end of the simulation.

On a second initialization stage, an event for each of the functions to call along the simulation is inserted into the queue, at their corresponding execution times. These initial events ensure a first execution of each function; successive executions will depend on each function returning an event for its next execution, which the simulator will insert in the priority queue.

The simulation starts by getting the first event in the priority queue, executes its function, and puts in the queue the new event returned by the function. Then again gets the first event in the queue, and repeats the process until the event labeled "EndSimulation" turns up. If a function does not return an event, it will not be executed again for the rest of the simulation. 

Typical events in PyWiSim include traffic generation, transmission, and resource assignment (schedulers). For very long simulations, an event "ShowProgress" may be included to show messages indicating the current execution time. 


## References

- Class PriorityQueue, in Python documentation, [queue - A synchronized queue class](https://docs.python.org/3/library/queue.html#queue.PriorityQueue).


[Back to README](../README.md)

