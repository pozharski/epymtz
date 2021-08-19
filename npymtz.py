import numpy as np
import gzip

def load(fname):
    try:
        dataset = np.load(fname)
    except ValueError:
        fgz = gzip.open(fname,'rb')
        dataset = np.load(fgz)
    return dataset

class dataset:
    def __init__(self, fname=None, data=None):
        if fname is not None:
            self.data = load(fname)
        elif data is not None:
            self.data = data
        else:
            self.data = None
    def __getitem__(self,*args,**kwds):
        return self.data.__getitem__(*args,**kwds)
    def __len__(self):
        return self.data.__len__()
    def set_data(self, data):
        self.data = data
        if '_hklash' in dir(self):
            del self._hklash
    def dbinned(self, Nperbin=1000, Nbins=None):
        data = np.sort(self.data,order='DSPACING')
        if Nbins is None:
            Nbins = int(np.floor(self.num_uniq()/Nperbin))
        return [dataset(data=dset) for dset in np.array_split(data,np.searchsorted(data['DSPACING'], [t[-1]['DSPACING'] for t in np.array_split(data,Nbins)[:-1]]))[::-1]]
    def hklbinned(self):
        return [[t2.binned_view('L') for t2 in t1.binned_view('K')] for t1 in  self.binned_view('H')]
    def binned_view(self, column):
        self.data.sort(order=column)
        return [dataset(data=dset) for dset in np.split(self.data,np.searchsorted(self.data[column],np.sort(np.unique(self.data[column])))[1:])]
    def hklash(self, force=False):
        if '_hklash' not in dir(self) or force:
            self._hklash = np.array(['%d_%d_%d' % (t['H'],t['K'],t['L']) for t in self.data])
        return self._hklash
    def num_uniq(self):
        return len(set(self.hklash()))
    def stats(self):
        return self.data['DSPACING'].max(), self.data['DSPACING'].min(), self.data['DSPACING'].mean(), len(self.data), self.num_uniq()
    def labels(self):
        return self.data.dtype.names
    def reftype(self):
        if 'FOSC' in self.data.dtype.names:
            return 'buster'
        elif 'FC_ALL' in self.data.dtype.names:
            return 'refmac'
        else:
            return 'unknown'
   
