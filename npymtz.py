import numpy as np
import gzip

def load(fname):
    try:
        dataset = np.load(fname)
    except ValueError:
        fgz = gzip.open(fname,'rb')
        dataset = np.load(fgz)
    return dataset
