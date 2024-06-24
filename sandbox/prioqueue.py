#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# prioqueue : tests on priority queues

'''Tests a discrete event simulator engine based on a priority queue.

Events are collected in a priority queue implemented as a Python data structure C{queue.PriorityQueue}. This data structure ensures efficient addition of events, and guarantees extraction in the correct order of precedence, i.e. earlier events first. Each event in the priority queue is a list::

    [ next_round, priority, id_action ]

where:
    - next_round : the next simulation round on which to execute the action indicated by this event.
    - priority: the order of execution among other possible actions to be executed on the same round. This ensures an order of execution for events scheduled to be executed on the same instance (the same round).
    - id_action: an identifier for the action to be executed.

Actions are recorded in a dictionary of the form::

    { id_action : action_function }

where:
    - C{id_action} : an identifier for the action to be executed.
    - C{action_function} : the method in an action class which executes a task.

Class C{Action} retains information on the task to be executed, namely an action identifier, a delay which adds to the current round number to indicate the next round in which this action is to be executed again, and the priority of execution for tasks to be executed on the same round. Method C{run()} in this class executes the task corresponding to this action. 

Class C{EventQueue} contains and processes the event priority queue, i.e. runs the simulation. It recognizes events by consulting a dictionary C{{id_action : action_function}} which returns the action method to execute according to the action identifier. Function C{run()} carries on the simulation until the total number of rounds have been executed.

The former organization allows to insert in this simulation engine any kind of task, provided it is coded in accordance to the C{Action} class structure.
'''


from queue import PriorityQueue
import sys

class Action():
    '''An action to execute when an event calls for it.
    '''

    def __init__(self, id_action, delay=1, priority=1):
        '''Constructor.

        @param id_action: an identifier for this action.
        @param delay: how many simulation interval units to skip before another execution of this action.
        @param priority: the order of execution for this action among other possible actions to be executed in the same instance.
        '''

        # action properties
        self.id_action = id_action
        '''An identifier for this action.'''
        self.delay = delay
        '''Intervals to skip before next execution.'''
        self.priority = priority
        '''Order of execution in a certain round.'''
        return
    

    def run(self, rn):
        '''Executes an action.

        @param rn: round number.
        @return: an event item [next_round, priority, id_action] to add to the priority queue.
        '''
        # execute action
        print("    {}: action:  {:12s}, delay {}, priority {}".\
              format(rn, self.id_action, self.delay, self.priority))
        # schedule next event
        event_item = [rn + self.delay, self.priority, self.id_action]
        return event_item



class EventQueue:
    '''A queue of events in priority order of execution.
    '''

    def __init__(self, dc_actions={}, nr_rounds=7):
        '''Constructor.

        @param dc_actions: a dictionary of {id_action : action_function}.
        @param nr_rounds: number of rounds to run the simulation, equivalent to simulation duration in simulation intervals.

        '''
        self.dc_actions = dc_actions    # dictionary of actions
        '''Dictionary of actions.'''
        self.nr_rounds = nr_rounds      # number of rounds to run
        '''Number of rounds to run the simulation.'''
        self.rn = 0                     # current round number
        '''Current round number.'''
        self.event_qu = PriorityQueue() # the event priority queue
        '''The event priority queue.'''

        # create and put in queue event to end simulation
        end_event = [self.nr_rounds+1, 1, "EndSimulation"]
        self.event_qu.put(end_event)
        return


    def run(self):
        '''Runs the simulation.

        Extracts events from the priority queue and executes them, until the number of rounds reaches the total number of rounds indicated for the simulation. Special action "Round" increments the round counter.
        '''

        while not self.event_qu.empty():
            event = self.event_qu.get()
            #print("current event", event)
            next_round, priority, id_action = event #event[2]
            if id_action == "EndSimulation":
                print("--- round {} simulation ended.".format(self.rn)) 
                break
            else:
                if self.rn == next_round:
                    pass
                else:
                    self.rn = next_round
                    print("--- round {}".format(self.rn)) 
            new_event = self.dc_actions[id_action](self.rn)
            self.event_qu.put(new_event)
        return



if __name__ == "__main__":
    '''Runs test.
    '''

    # create actions dictionary
    dc_actions = {}

    # list of actions: [id_action, delay, priority]
    ls_tasks = [ \
        ["TransmitTBs", 0.5, 2], \
        ["GenTraffic", 0.7, 3], \
        ["GenerateTBs", 0.5, 4], \
        ["AssignRes", 4, 5], \
        ]
        #["Round", 1, 1], \

    # add actions to dictionary
    for task in ls_tasks:
        id_action, delay, priority = task
        ac_trtbs = Action(id_action, delay, priority)
        dc_actions[id_action] = ac_trtbs.run
    
    # create priority queue
    ev_qu = EventQueue(dc_actions, 11)

    # execute initial actions, to generate a first event of each type
    for task in ls_tasks:
        id_action, delay, priority = task
        ev_qu.event_qu.put([delay, priority, id_action])

    # run the simulation
    ev_qu.run()

