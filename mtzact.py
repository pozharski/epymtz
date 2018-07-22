#! /usr/bin/env python3

import os

headerhelp = \
'''
Run MTZ actions.

Detains on actions:

testset             Generate a test set.
rcompute            Compute R-values.
rotate_freeflag     Rotate test set flags.
'''
from argparse import ArgumentParser, RawDescriptionHelpFormatter
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                        description=headerhelp)
parser.add_argument('-a', '--action', action='append',
                    choices = [	'testset',
                                'rcompute',
                                'rotate_freeflag',
                                ],
                    default = [],
                    metavar = '', help='Action to perform')
parser.add_argument('--folder', default=os.getcwd(), help='Folder to work in, defaults to CWD')
parser.add_argument("-m", "--mtzin", help="Input MTZ file name.")
parser.add_argument("-o", "--mtzout", help="Output MTZ file name.")
parser.add_argument("--free-fraction", default=0.05, type=float, help='Fraction of test reflections')
parser.add_argument("--dhigh", type=float, help='High resolution cutoff.')
parser.add_argument("--dlow", type=float, help='Low resolution cutoff.')
parser.add_argument("--free", default='FreeR_flag', help='Column label for freeflag.')
parser.add_argument("--fobs", default='F', help='Column label for Fobs.')
parser.add_argument("--fcalc", default='FC_ALL', help='Column label for Fcalc.')
parser.add_argument("--sigfobs", default='SIGF', help='Column label for sigmaF.')
parser.add_argument("-f", "--force", action="store_true", help='Force overwriting, no questions asked.  Dangerous.')
parser.add_argument('--label-style', choices = ['refmac','phenix','buster','pdbredo'], help='Style of output column names')
parser.add_argument('--free-shift', default=1, type=int, help='Amplitude of shift when rotating test set')

args = parser.parse_args()

args.folder = os.path.abspath(args.folder)

if args.label_style:
    if args.label_style == 'refmac':
        args.free = 'FreeR_flag'
        args.fcalc = 'FC_ALL'
    elif args.label_style == 'phenix':
        args.free = 'R-free-flags'
        args.fcalc = 'F-model'
        args.fobs = 'F-obs-filtered'
        args.sigfobs = 'SIGF-obs-filtered'
    elif args.label_style == 'buster':
        args.fcalc = 'FC'
        args.fobs = 'FOSC'
        args.sigfobs = 'SIGFOSC'
    elif args.label_style == 'pdbredo':
        args.free = 'FREE'
        args.fobs = 'FP'
        args.sigfobs = 'SIGFP'
        args.fcalc = 'FC_ALL'

import mtzactions

for action in args.action:
    mtzactions.__getattribute__(action)(args)

