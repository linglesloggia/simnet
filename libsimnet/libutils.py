#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# libutils : utilities library
#
'''PyWiSim utilities library.
'''

from threading import Lock
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



def run_qa_tests(ls_tests, par_op=None, fn_par=None):
    '''Runs tests from a list of (message, test_function).

    @param ls_tests: a list of tuples (message, test_function).
    @param par_op: number of test to run, default None if not given.
    @param fn_par: a parameter to pass to test_function, default None.
    @return: user's input option, for further actions, or "q" if input option is nor valid.
    '''
    if par_op != None:
        op = par_op
    else:
        # show options to run functions
        for i in range(0, len(ls_tests)):
            msg = "  {} " + ls_tests[i][0]
            print(msg.format(i+1))
        #op = input("  Test to run (0 for all): ")
        op = input("  Test to run): ")
    # execute function(s)
    try:
        iop = int(op)
    except:
        return "q"
    if iop == 0:    # not always working
        for i in range(0, len(ls_tests)):
            print("\n=== {}".format(ls_tests[i][0]))
            if len(ls_tests[i]) == 3:
                fn_par = ls_tests[i][2]
                ls_tests[i][1](fn_par)  # executes function with parameter
            else:
                ls_tests[i][1]()        # executes function without parameter
    elif iop > 0 and iop <= len(ls_tests):
        print("\n=== {}".format(ls_tests[iop-1][0]))
        if len(ls_tests[iop-1]) == 3:
            fn_par = ls_tests[iop-1][2]
            ls_tests[iop-1][1](fn_par)  # executes function with parameter
        else:
            ls_tests[iop-1][1]()        # executes function without parameter
    else:
        return "q"
    return op

