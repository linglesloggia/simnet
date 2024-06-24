#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''PyWiSim: a system level wireless network simulator and framework.

Package structure:
    - C{libsimnet} : the main library, where base classes reside; this directory is assumed to be left untouched in an extension; it is not intended to be modified.
    - C{extensions} : contains the code for the extension, in a subdirectory with the extension name, e.g. C{myextension}. The extension code may overwrite the libsimnet classes, as well as add new classes or functions, or import from external libraries. Tests for the extension are also expected to be included in this directory.
    - C{models} : contains the code for simulation models for different specific entities, such as channel, scheduler, traffic generator, or other. Classes defined here may also be imported into an extension based on the PyWiSim simple example, or to a totally new extension. 
'''
