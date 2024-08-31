#
# treadstest: a performance test of threads versus lists
#

'''Test performance of execution in threads versus in list.

Two classes of objects are defined, both implementing the same function, but one runs the function in its own thread (the class inherits from Thread), while the function in the other class will be called from outside. Two lists are built, one list with objects inheriting from Thread, the other with objects not inheriting from Thread. The first list runs a thread in each object, the other list runs only one thread and calls the function on each object in the list. Performance of the two implementations are compared in terms of processing time.

TODO: sleep time supressed; probably not necessary for this test.

Suggested tests::

    $ python3 threadstest.py 20 0.001 20 0
    $ python3 threadstest.py 20 0.001 20 0
    $ python3 threadstest.py 200 0.001 200 0
    $ python3 threadstest.py 1000 0.001 1000 0
    $ python3 threadstest.py 2000 0.001 2000 0
'''

from threading import Thread, Lock
from time import perf_counter, sleep
import sys


lock_obj = Lock()
'''To lock thread for mutually exclusive printing.'''

def mutex_prt(msg):
    '''Mutually exclusive printing.

    @param msg: string to print.
    '''
    lock_obj.acquire()
    print(msg)
    lock_obj.release()
    return


def fn(n=100):
    '''A function just to demand some processing.
    '''
    for i in range(0, n):
        y = i * i
        z = y * i
    return


class EntityThread(Thread):
    '''A class with its own execution thread.
    '''
    def __init__(self, id_obj, sim_step, nr_rounds, debug=0):
        Thread.__init__(self)
        self.id_obj = id_obj
        self.sim_step = sim_step
        self.nr_rounds = nr_rounds
        self.debug = debug
        return
    def run(self):
        for rn in range(0, nr_rounds):
            fn()
            if self.debug == 2:
                print("   ..." + self.id_obj + ", round " + str(rn) )
            #sleep(self.sim_step)
        if self.debug >= 1:
            mutex_prt("{:10s} finished, perf_counter {:10.3f}".\
                  format(self.id_obj, perf_counter() ) )
        return


class EntityItem():
    '''A class for objects as items in a list run by a single thread.
    '''
    def __init__(self, id_obj, debug=0):
        Thread.__init__(self)
        self.id_obj = id_obj
        self.sim_step = sim_step
        self.nr_rounds = nr_rounds
        self.debug = debug
        return
    def run_fn(self, rn):
        fn()
        if self.debug == 2:
            print("      ..." + self.id_obj + ", round " + str(rn) )
        return


def fn_ethreads(debug=0):
    '''A function to run the threads in each object.
    '''
    if debug:
        print("--- Objects in threads, one thread per object ")
    ls_objs = []
    start_time = perf_counter()
    for i in range(0, nr_objects):
        ls_objs += [ \
            EntityThread("ethread_"+str(i), sim_step, nr_rounds, debug) ]
    for obj in ls_objs:
        obj.start()
    for obj in ls_objs:
        obj.join()
    stop_time = perf_counter()
    print("    Objects in thread, simulation time: {:8.3f} secs".\
          format(stop_time - start_time))
    return


def fn_elist_run(ls_objs, nr_rounds, debug):
    '''A function to call the functions in objects from only one thread.
    '''
    for rn in range(0, nr_rounds):    # simulation rounds
        for obj in ls_objs:             # exec fn for objects in list
            obj.run_fn(rn)
        if debug >= 1:
            print("   ...round", rn, "finished")
        #sleep(sim_step)
    return


def fn_elist(debug=0):
    '''A function to run the thread which calls the functions in objects.
    '''
    if debug:
        print("--- Objects in list, a single thread ")
    ls_objs = []
    for i in range(0, nr_objects):
        ls_objs += [EntityItem("eitem_"+str(i), debug)]

    thrlst = Thread(target=fn_elist_run, args=(ls_objs,nr_rounds,debug,) )
    start_time = perf_counter()
    thrlst.start()
    thrlst.join()
    stop_time = perf_counter()
    print("    Objects in list, simulation time: {:10.3f} secs".\
          format(stop_time - start_time))
    return


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Run with 10 objects, step 0.1 secs, 10 rounds, debug 3")
        print("")
        print("For options,")
        print("    python3 threadtest.py -h")
        print()
        nr_objects = 10
        sim_step = 0.1
        nr_rounds = 10
        debug = 3
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Use:")
        print("    python3 [nr_objects sim_step nr_rounds debug")
        print("  nr_objects; number of objects to create")
        print("  sim_step: time in seconds for simulation step")
        print("  nr_rounds: number of times to run the simulation")
        print("  debug: 0, 1 or 2; 0 for no messages")
        sys.exit()
    elif len(sys.argv) > 1:
        nr_objects = int(sys.argv[1])
        sim_step = float(sys.argv[2])
        nr_rounds = int(sys.argv[3])
        debug = int(sys.argv[4])
    print("Simulation run with threads and list")
    print("    nr_objects={:d}, sim_step={:.3f}, nr_rounds={:d}".\
          format(nr_objects, sim_step, nr_rounds))
    if nr_objects > 100 or nr_rounds > 100: 
        print("May take a time, please wait...")
    #sys.exit()
        
    fn_ethreads(debug)
    fn_elist(debug)



