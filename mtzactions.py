from pymtz import read_mtz_file
import sys, os
from scipy import array, isfinite, argsort, cumsum
from matplotlib.pyplot import show, grid, figure, title, plot, cla, xlabel, ylabel
import utils

def testset(args):
    mtz = read_mtz_file(args.mtzin)
    sys.stdout.write('Generating the free flag column (%.1f%%)... ' % (100*args.free_fraction))
    mtz.GenerateFreeFlag(args.free, args.free_fraction)
    sys.stdout.write('done\n')
    utils.mtzsave(mtz, args)

def rotate_freeflag(args):
    mtz = read_mtz_file(args.mtzin)
    avlabels = mtz.GetLabels()
    if args.free not in avlabels:
        print('Column '+args.free+' not found. Check input mtz-file and test set label selection.')
        print('Available columns:')
        print(' '.join(avlabels))
        return False
    sys.stdout.write('Rotating the free flag column by %d... ' % (args.free_shift))
    mtz.RotateFreeFlag(args.free, args.free_shift)
    sys.stdout.write('done\n')
    utils.mtzsave(mtz, args)

def rcompute(args):
    mtz = read_mtz_file(args.mtzin)
    if not args.dhigh:
        args.dhigh = mtz.GetHighD()
    avlabels = mtz.GetLabels()
    if args.free not in avlabels:
        print('Column '+args.free+' not found. Check input mtz-file and test set label selection.')
        print('Available columns:')
        print(' '.join(avlabels))
    if args.fobs not in avlabels:
        print('Column '+args.fobs+' not found. Check input mtz-file and Fobs label selection.')
        print('Available columns:')
        print(' '.join(avlabels))
    if args.fcalc not in avlabels:
        print('Column '+args.fcalc+' not found. Check input mtz-file and Fcalc label selection.')
        print('Available columns:')
        print(' '.join(avlabels))
    else:
        d = array(mtz.GetResolutionColumn())
        fp, fc, sigf = mtz.GetReflectionColumns([args.fobs, args.fcalc, args.sigfobs])
        test_index = mtz.GetTestIndex(args.free, True)
        work_index = mtz.GetWorkIndex(args.free, True)
        ind = (d >= args.dhigh) * isfinite(fp)
        test_index = test_index[ind]
        work_index = work_index[ind]
        fp, fc, sigf = fp[ind], fc[ind], sigf[ind]
        fpfc = fp-fc
        sys.stdout.write('%9s %9s %9s %9s\n' % ('R','RWORK','RFREE','REXP'))
        sys.stdout.write('%9.3f '  % (100*abs(fpfc).sum()/fp.sum()))
        sys.stdout.write('%9.3f '  % (100*abs(fpfc[work_index]).sum()/fp[work_index].sum()))
        sys.stdout.write('%9.3f ' % (100*abs(fpfc[test_index]).sum()/fp[test_index].sum()))
        sys.stdout.write('%9.3f\n '  % (79.8*sigf.sum()/fp.sum()))
        dR=(abs(fpfc)*fp.sum()-abs(fpfc).sum()*fp)/((fp.sum())**2-fp*fp.sum())
        indr = argsort(dR)
        Rc = 100*cumsum(abs(fpfc[indr]))/cumsum(fp[indr])
        Rce = 79.8*cumsum(sigf[indr])/cumsum(fp[indr])
        plot(Rc-Rce,'r')
        grid()
        show()
        
