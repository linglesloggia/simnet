#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#


'''Race condition example.

Different threads add a letter to a common string. Several runs of the code give different results according to which thread finishes first. Name of thread is the letter it adds.

Adapted from: https://www.pythontutorial.net/python-concurrency/python-threading-lock/ 
'''

from threading import Thread, Lock
from time import sleep

lock = Lock()
counter = "" 

def increase_lock(by, id_thread, lck):
    global counter
    if lck:
        lock.acquire()
    local_counter = counter
    local_counter += id_thread
    sleep(0.1)
    counter = local_counter
    print("{} : {:5s} |  ".format(id_thread, counter.rjust(4)), end="")
    if lck:
        lock.release()
    return

def exec_increase(rounds, lck):
    global counter
    for i in range(0, rounds):
        # create threads
        t1 = Thread(target=increase_lock, args=("A", "A", lck))
        t2 = Thread(target=increase_lock, args=("B", "B", lck))
        t3 = Thread(target=increase_lock, args=("C", "C", lck))
        t4 = Thread(target=increase_lock, args=("D", "D", lck))
        counter ="" 
        # start the threads
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        # wait for the threads to complete
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        print(" final: {}".format(counter))

print("=== Several threads add a letter to a string")
print("Shows thread_id : string")
print("--- no lock")
exec_increase(10, False)

print("--- lock")
exec_increase(10, True)
