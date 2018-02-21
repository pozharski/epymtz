#! /usr/bin/env python

import sys, os

from scipy.stats import nanmean
from scipy import array

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-m", "--mtzin", 
                dest="mtzin",
                help="Input MTZ file name.")
parser.add_option("-o", "--mtzout", 
                dest="mtzout",
                help="Output MTZ file name.")
parser.add_option("--generate-test-set", 
                dest='genfree', 
                default=False, 
                action="store_true",
                help='Generate a free-flag column.')
parser.add_option("--free-fraction", 
                dest='freefrac', 
                default=0.05, 
                type=float,
                help='Fraction of test reflections')
parser.add_option("--add-noise", 
                dest='noiselevel', 
                default=0.0, 
                type=float,
                help='Add noise based on sigmas.')
parser.add_option("--fobs",
                dest='fobs',
                default='FP',
                help='Column label for Fobs.')
parser.add_option("--sigfobs",
                dest='sigfobs',
                default='SIGFP',
                help='Column label for sigmaF.')
parser.add_option("-f", "--force",
                dest='force',
                default=False, 
                action="store_true",
                help='Force overwriting, no questions asked.  Dangerous.')
parser.add_option("--fcalc",
                dest='fcalc',
                default='FC_ALL',
                help='Column label for Fcalc.')
parser.add_option("--free",
                dest='free',
                default='FreeR_flag',
                help='Column label for freeflag.')
parser.add_option("--r-values", 
                dest='rprint', 
                default=False, 
                action="store_true",
                help='Print the Rall/Rwork/Rfree.')
parser.add_option("--col-stats", 
                dest='colprint', 
                default=False, 
                action="store_true",
                help='Print column statistics.')
parser.add_option("--test-size",
                dest='freeprint', 
                default=False, 
                action="store_true",
                help='Print test set size.')
parser.add_option("--dhigh", 
                dest='dhigh', 
                type=float,
                help='High resolution cutoff.')
parser.add_option("--dlow", 
                dest='dlow', 
                type=float,
                help='Low resolution cutoff.')
parser.add_option("--cell",
                dest='cell',
                help='Reset unit cell parameters.')
parser.add_option("--shuffle-thin-frames",
                dest='shuffle_thin_frames',
                default=False, 
                action="store_true",
                help='Shuffle reflections in thin frames.')


(options, args) = parser.parse_args()

if options.mtzin:
    mtzin=options.mtzin
else:
    mtzin=args[0]
    if not mtzin:
        print "mtztrick -h for help"
        sys.exit()

from pyshakerr.pymtz import read_mtz_file
mtz = read_mtz_file(mtzin)

if not options.dhigh:
    options.dhigh = mtz.GetHighD()

if not options.dlow:
    options.dlow = mtz.GetLowD()

if options.rprint:
    if options.free not in mtz.GetLabels():
        print 'Column '+options.free+' not found. Check input mtz-file and test set label selection.'
    else:
        d = array(mtz.GetResolutionColumn())
        fp, fc = mtz.GetReflectionColumns([options.fobs, options.fcalc])
        test_index = mtz.GetTestIndex(options.free)
        test_index = test_index[d[test_index] > options.dhigh]
        work_index = mtz.GetWorkIndex(options.free)
        work_index = work_index[d[work_index] > options.dhigh]
        fpfc = fp-fc
        sys.stdout.write('%9.3f '  % (100*nanmean(abs(fpfc[d>options.dhigh]))/nanmean(fp[d>options.dhigh])))
        sys.stdout.write('%9.3f '  % (100*nanmean(abs(fpfc)[work_index])/nanmean(fp[work_index])))
        sys.stdout.write('%9.3f\n' % (100*nanmean(abs(fpfc)[test_index])/nanmean(fp[test_index])))

if options.colprint:
    sys.stdout.write('Resolution limits %7.2f - %7.2f\n' % (mtz.GetLowD() ,mtz.GetHighD()))
    sys.stdout.write('              Column  present  missing Comp,%\n')
    for label in mtz.GetLabels():
        sys.stdout.write('%20s %8d %8d %6.2f\n' % (label,
                                            mtz.ColumnPresentNumber(label),
                                            mtz.ColumnMissingNumber(label),
                                            100*mtz.ColumnCompleteness(label)))

if options.freeprint:
    if options.free not in mtz.GetLabels():
        print 'Column '+options.free+' not found. Check input mtz-file and test set label selection.'
    else:
        print '--------------------'
        print 'Test set information'
        print 'Label:    '+options.free
        info = mtz.GetTestInfo(options.free)
        if info:
            print 'Style:    %s\nFraction: %.2f\nSize:     %d' % info
        else:
            print 'Test set style unknown'
        print '--------------------'

if options.genfree:
    sys.stdout.write('Generating the free flag column (%.1f%%)... ' % (100*options.freefrac))
    mtz.GenerateFreeFlag(options.free, options.freefrac)
    sys.stdout.write('done\n')

if options.noiselevel > 0:
    sys.stdout.write('Randomizing ' + options.fobs + ' column using ' + 
                        str(options.noiselevel) + ' of column ' + 
                        options.sigfobs + ' ...')
    mtz.ShakeColumn(fp_label=options.fobs, 
                    sigfp_label=options.sigfobs,
                    RSF=options.noiselevel)
    sys.stdout.write('done\n')

if options.cell:
    for dset in range(mtz.GetDatasetNumber()):
        mtz.SetDatasetCell(dset, tuple(map(float,options.cell.split(','))))

if options.shuffle_thin_frames:
    sys.stdout.write('Shuffling columns ' + options.fobs + '/' + 
                        options.sigfobs + ' in thin frames... ')
    sys.stdout.write('actual resolution range %.2f-%.2f... done\n' % mtz.shell_shuffle(labels=[options.fobs, options.sigfobs], num=100, d_low=options.dhigh))

if options.mtzout:
    if os.access(options.mtzout, os.F_OK):
        if options.force:
            sys.stdout.write('Overwriting MTZ file (you did set --force flag)\n')
        else:
            sys.stdout.write('Overwrite MTZ file? Enter "yes" to confirm ')
            if raw_input().lower() != 'yes':
                sys.stdout.write('Phew... that was close...\n')
                sys.exit(1)
            else:
                sys.stdout.write('Well, if you insist...\n')
    sys.stdout.write('Writing the output MTZ file '+options.mtzout + '... ')
    mtz.write(options.mtzout)
    sys.stdout.write('done\n')

