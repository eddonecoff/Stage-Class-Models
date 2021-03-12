"""
hw4.py
Name(s):
NetID(s):
Date:
"""

import numpy
import matplotlib.pyplot as plt

def longTime(A):
    return (domEig, domVec)

def getAk(A,k):
    return AK

def error(uCurr, uLong):
    return err

def normalize(v):
    return v

def simulate(A,k,u0):
    return sim

"""
main function
"""
if __name__ == '__main__':

    # Scenario A: Frogs
    frogMat = [[0.0, 0.0, 3.0,  8.0], \
               [0.4, 0.0, 0.0,  0.0], \
               [0.0, 0.5, 0.0,  0.0], \
               [0.0, 0.0, 0.25, 0.0]]
    frogInit = [[0.0],[0.0],[0.0],[250.0]]

    # Scenario B: Owls
    owlMat = [[0.2,  0.1,  0.4,  1/3], \
              [0.4,  0.4,  0.2,  1/3], \
              [0.2,  0.3,  0.2,  1/3], \
              [0.01, 0.01, 0.01, 1.5]]
    owlInit = [[100.0],[100.0],[0.0],[0.0]]
