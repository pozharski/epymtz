#! /usr/bin/env python3

# There should be a way to do this with relative imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.figure import Figure
from matplotlib.pyplot import grid
class KDWindow(Figure):
    def __init__(self, *args, **kwds):
        Figure.__init__(self, *args, **kwds)
    def set_data(self, data):
        self.Ish = data
    def plot(self):
        self.add_subplot(111)
        self.axes[0].clear()
        self.pt = self.axes[0].plot(self.Ish[19][0],self.Ish[19][1])
        grid(True)
        self.canvas.draw()

def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    headerhelp = \
'''
'''
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=headerhelp)
    parser.add_argument('inpath',
                        help='The input mTZ file.')
    parser.add_argument('-n', '--nobs-per-shell', type=int, default=1000,
                        help='Number of observations per shaell target.')
    
    args = parser.parse_args()

    from pymtz import read_mtz_file
    from scipy import random, corrcoef, array, nonzero, zeros, ones
    from scipy.stats import spearmanr
    from matplotlib.pyplot import figure, show
    
    print("Reading ",args.inpath, end='...', flush=True)
    dataset = read_mtz_file(args.inpath)
    print("done")
    Nmeas = dataset.GetReflectionNumber()
    print("Found %d measurements" % Nmeas)
    ashes = dataset.GetHKLhash()
    uniqhkl = set(ashes)
    Nunq = len(uniqhkl)
    Nshells = int(Nunq/args.nobs_per_shell)
    Ihl,shl=array(dataset.reflections).T[nonzero([(t in ['I', 'SIGI']) for t in dataset.GetLabels()])[0]]
    whl=1/shl**2
    hkldict = dict.fromkeys(uniqhkl)
    for (i,ash) in enumerate(ashes):
        if hkldict[ash]:
            hkldict[ash].append(i)
        else:
            hkldict[ash] = [i]
    print("Found %d unique reflections" % (Nunq))
    print("Will subdivide data into %d resolution shells" % Nshells)
    edges = dataset.GetResolutionShells(Nshells)
    shell_column = dataset.GetShellColumn(edges)
    cc12, sp12, p12, Ish = [], [], [], []
    for i in range(Nshells):
        Ish1, Ish2 = [], []
        shind = (shell_column == i)
        for ash in  set(ashes[shind]):
            mcity = len(hkldict[ash])
            if mcity > 1:
                Nhalf = int(mcity/2)
                wr = list(ones(Nhalf))+list(zeros(Nhalf))
                if len(wr)<mcity:
                    wr.append(random.random_integers(0,1))
                wr = random.permutation(wr)
                Ish1.append(sum(wr*Ihl[hkldict[ash]]*whl[hkldict[ash]])/sum(wr*whl[hkldict[ash]]))
                Ish2.append(sum((1-wr)*Ihl[hkldict[ash]]*whl[hkldict[ash]])/sum((1-wr)*whl[hkldict[ash]]))
        cc12.append(corrcoef(Ish1, Ish2)[0][1])
        sp12t, p12t = spearmanr(Ish1, Ish2)
        sp12.append(sp12t)
        p12.append(p12t)
        Ish.append([Ish1,Ish2])
        print("%4d %7.2f %7.2f %7.2f %8d %5.3f %5.3f %6.3g" % (i+1, edges[i], edges[i+1], dataset.GetResolutionColumn()[shind].mean(), sum(shind), cc12[-1], sp12t, p12t))
    Ish_fig = figure(FigureClass=KDWindow)
    Ish_fig.set_data(Ish)
    Ish_fig.plot()
    show()

if __name__ == "__main__":
    main()
