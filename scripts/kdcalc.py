#! /usr/bin/env python3

# There should be a way to do this with relative imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.figure import Figure
from matplotlib.pyplot import grid
from scipy.stats import rankdata
from scipy import sqrt

class KDWindow(Figure):
    def __init__(self, *args, **kwds):
        Figure.__init__(self, *args, **kwds)
        self.pt={}
    def set_data(self, data):
        self.Ish, self.d12, self.cc12, self.sp12, self.corecc12 = data
        self.target = 0
    def plot(self):
        self.add_subplot(231)
        self.axes[0].clear()
        self.pt['cc12'] = self.axes[0].plot(1/self.d12**2,self.cc12,'bo',1/self.d12[self.target]**2,self.cc12[self.target]+0.1,'ro')
        vals = 1/sqrt(self.axes[0].get_xticks())
        self.axes[0].set_xticklabels(("%.2f "*len(vals) % tuple(vals)).split())
        self.axes[0].set_title("%.2f / %.3f" % (self.d12[self.target], self.cc12[self.target]))
        grid(True)
        self.add_subplot(232)
        self.axes[1].clear()
        self.pt['sp12'] = self.axes[1].plot(1/self.d12**2,self.sp12,'go',1/self.d12[self.target]**2,self.sp12[self.target]+0.1,'ro')
        vals = 1/sqrt(self.axes[1].get_xticks())
        self.axes[1].set_xticklabels(("%.2f "*len(vals) % tuple(vals)).split())
        self.axes[1].set_title("%.2f / %.3f" % (self.d12[self.target], self.sp12[self.target]))
        grid(True)
        self.add_subplot(233)
        self.axes[2].clear()
        self.pt['corecc12'] = self.axes[2].plot(1/self.d12**2,self.corecc12,'yo',1/self.d12[self.target]**2,self.corecc12[self.target]+0.1,'ro')
        vals = 1/sqrt(self.axes[2].get_xticks())
        self.axes[2].set_xticklabels(("%.2f "*len(vals) % tuple(vals)).split())
        self.axes[2].set_title("%.2f / %.3f" % (self.d12[self.target], self.corecc12[self.target]))
        grid(True)
        self.add_subplot(234)
        self.axes[3].clear()
        self.pt['ish'] = self.axes[3].plot(self.Ish[self.target][0],self.Ish[self.target][1],'b.')
        grid(True)
        self.add_subplot(235)
        self.axes[4].clear()
        self.pt['ranked'] = self.axes[4].plot(rankdata(self.Ish[self.target][0]),rankdata(self.Ish[self.target][1]),'g.')
        grid(True)
        self.add_subplot(236)
        self.axes[5].clear()
        zind = self.Ish[self.target][2]
        self.pt['ish'] = self.axes[5].plot(self.Ish[self.target][0][zind],self.Ish[self.target][1][zind],'y.')
        grid(True)
        self.canvas.draw()
    def onkeypress(self, event):
        if event.key == 'right':
            self.target = min(self.target+1, len(self.d12)-1)
            self.plot()
        if event.key == 'left':
            self.target = max(self.target-1, 0)
            self.plot()

def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    headerhelp = \
'''
'''
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=headerhelp)
    parser.add_argument('inpath',
                        help='The input MTZ/NPYMTZ file.')
    parser.add_argument('-n', '--nobs-per-shell', type=int, default=1000,
                        help='Target number of observations per shell target.')
    parser.add_argument('-b', '--nbins', type=int,
                        help='Number of resolution shelld, supersedes number of observations per shell target.')
    parser.add_argument('--zcutoff', type=float, default=3.0,
                        help='Z-score cutoff for core CC calculation.')
    
    args = parser.parse_args()

    from pymtz import read_npymtz_file
    from scipy import random, corrcoef, array, nonzero, zeros, ones
    from scipy.stats import spearmanr, iqr
    from matplotlib.pyplot import figure, show
    import numpy as np
    
    print("Reading ",args.inpath, end='...', flush=True)
    dataset = read_npymtz_file(args.inpath)
    print("done")
    print("Found %d unique reflections" % (dataset.num_uniq()))
    z = dataset.dbinned(Nperbin=args.nobs_per_shell, Nbins=args.nbins)
    print("Will subdivide data into %d resolution shells" % len(z))
    d12, cc12, sp12, p12, Ish, corecc12 = [], [], [], [], [], []
    print("Num      D1      D2")
    for i,zs in enumerate(z):
        Ish1, Ish2 = [], []
        d1,d2,dmean,Nmeas,Nuniq = zs.stats()
        d12.append(dmean)
        for Ikl in zs.hklbinned():
            for Il in Ikl:
                for dset in Il:
                    if len(dset) > 1:
                        half1,half2 = np.array_split(np.random.permutation(dset.data),2)
                        I1,whl1,I2,whl2 = half1['I'],1/half1['SIGI']**2,half2['I'],1/half2['SIGI']**2
                        Ish1.append(sum(I1*whl1)/sum(whl1))
                        Ish2.append(sum(I2*whl2)/sum(whl2))
        cc12.append(corrcoef(Ish1, Ish2)[0][1])
        sp12t, p12t = spearmanr(Ish1, Ish2)
        sp12.append(sp12t)
        p12.append(p12t)
        Ish1, Ish2 = np.array(Ish1), np.array(Ish2)
        zind = (abs(Ish1-np.median(Ish1))/iqr(Ish1,scale='normal')<args.zcutoff)*(abs(Ish2-np.median(Ish2))/iqr(Ish2,scale='normal')<args.zcutoff)
        deltish = Ish2-np.polyval(np.polyfit(Ish1,Ish2,1),Ish1)
        zind *= abs(deltish-np.median(deltish))/iqr(deltish,scale='normal')<args.zcutoff
        Ish.append([Ish1,Ish2,zind])
        corecc12.append(corrcoef(Ish1[zind],Ish2[zind])[0][1])
        print("%4d %7.2f %7.2f %7.2f %8d %8d %5.3f %5.3f %10.3g %5.3f" % (i+1,d1,d2,dmean,Nmeas,Nuniq, cc12[-1], sp12t, p12t, corecc12[-1]))
    Ish_fig = figure(FigureClass=KDWindow)
    Ish_fig.set_data([Ish,array(d12),cc12,sp12,corecc12])
    Ish_fig.plot()
    Ish_fig.canvas.mpl_connect('key_press_event', Ish_fig.onkeypress)
    show()

if __name__ == "__main__":
    main()
