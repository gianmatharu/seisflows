import numpy as np 

def load(file):
    arr = np.fromfile(file, dtype='float32')
    return arr

vp = load('vp.bin')
vs = load('vs.bin')

ratio = vp/vs


