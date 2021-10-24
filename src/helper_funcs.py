import carla
import sys
import numpy as np
import random

def printl(L, log_name=""):
    if log_name: print(log_name)
    for idx, item in enumerate(L):
        print(idx, item)
    return