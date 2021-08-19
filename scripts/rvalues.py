#! /usr/bin/env python3

# There should be a way to do this with relative imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def rvalue(f1, f2):
    return 100*(abs(f1-f2).sum())/f1.sum()

def rvalues(f1, f2, ind, indf):
    ind2 = ind * indf
    return 100*(abs(f1[ind]-f2[ind]).sum())/f1[ind].sum(), 100*(abs(f1[ind2]-f2[ind2]).sum())/f1[ind2].sum()
    
def tuplediff(t1, t2):
    return tuple(np.array(t2)-np.array(t1))

def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    headerhelp = \
'''
'''
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=headerhelp)
    parser.add_argument('inpath',
                        help='The input MTZ/NPYMTZ file.')
    args = parser.parse_args()

    from pymtz import read_npymtz_file

    print("Reading ",args.inpath, end='...', flush=True)
    dataset = read_npymtz_file(args.inpath)
    print("done")

    if dataset.reftype() == 'buster':
        data = dataset[np.isfinite(dataset['F'])]
        Fo = data['FOSC']
        Fc = data['FC']
        test_set = data['FreeR_flag']
        h, k, l = data['H'], data['K'], data['L']
        foversigma = data['F']/data['SIGF']
    elif dataset.reftype() == 'refmac':
        data = dataset[np.isfinite(dataset['FP'])]
        Fo = data['FP']
        Fc = data['FC_ALL']
        test_set = data['FREE']
        h, k, l = data['H'], data['K'], data['L']
        foversigma = data['FP']/data['SIGFP']
    else:
        print("Unknown MTZ file type")
        sys.exit(1)
    
    print("R-values")
    indf = test_set == 0
    print("Regular:            %5.2f %% / %5.2f %%" % (rvalue(Fo, Fc), rvalue(Fo[indf], Fc[indf])))
    ind = h % 2 == 0
    print("H_even:             %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = h % 2 == 1
    print("H_odd:              %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = k % 2 == 0
    print("K_even:             %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = k % 2 == 1
    print("K_odd:              %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = l % 2 == 0
    print("L_even:             %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = l % 2 == 1
    print("L_odd:              %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    print("F/sigF cutoff")
    for cutoff in range(1,11):
        ind = foversigma>cutoff
        print("              %2d:    %5.2f %% / %5.2f %%" % (tuple([cutoff])+rvalues(Fo, Fc, ind, indf)))
    print("Resolulion cutoff")
    ind = data['DSPACING'] > 4
    print("               4.0A: %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = data['DSPACING'] > 3.5
    print("               3.5A: %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = data['DSPACING'] > 3
    print("               3.0A: %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = data['DSPACING'] > 2.5
    print("               2.5A: %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))
    ind = data['DSPACING'] > 2.0
    print("               2.0A: %5.2f %% / %5.2f %%" % rvalues(Fo, Fc, ind, indf))

    indh1 = h % 2 == 0
    indh2 = h % 2 == 1
    indk1 = k % 2 == 0
    indk2 = k % 2 == 1
    indl1 = l % 2 == 0
    indl2 = l % 2 == 1
    for D in [5,4.5,4,3.5,3,2.5,2,1.5]:
        indr = data['DSPACING'] > D
        R0, Rf0 = rvalues(Fo, Fc, indr, indf)
        print("%4.1f" % (D), end=' ')
        print("%4.1f/%4.1f" % (R0, Rf0), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indh1*indr, indf), (R0,Rf0)), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indh2*indr, indf), (R0,Rf0)), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indk1*indr, indf), (R0,Rf0)), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indk2*indr, indf), (R0,Rf0)), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indl1*indr, indf), (R0,Rf0)), end=' ')
        print("%4.1f/%4.1f" % tuplediff(rvalues(Fo, Fc, indl2*indr, indf), (R0,Rf0)))

if __name__ == "__main__":
    main()
