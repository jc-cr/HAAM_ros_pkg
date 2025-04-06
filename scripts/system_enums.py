#!/usr/bin/env python3
# system_enums.py
from enum import Enum

class SystemState(Enum):
    UNINITIALIZED = 0
    HOST_READING = 1
    SCANNING = 2
    STOPPED = 3
    WARNING = 4
    IDLE = 5
    ACTIVE = 6

class HostState(Enum):
    UNDEFINED = 0
    STOPPED = 1
    STARTING = 2
    IDLE = 3
    SUSPENDED = 4
    EXECUTE = 5
    STOPPING = 6
    ABORTING = 7
    ABORTED = 8
    HOLDING = 9
    HELD = 10
    RESETTING = 11
    SUSPENDING = 12
    UNSUSPENDING = 13
    CLEARING = 14
    UNHOLDING = 15
    COMPLETING = 16
    COMPLETE = 17