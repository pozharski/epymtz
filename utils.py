import sys, os
def mtzsave(mtz, args):
    '''
    Standard loop for writing mtz output.
    '''
    if os.access(args.mtzout, os.F_OK):
        if args.force:
            sys.stdout.write('Overwriting MTZ file (you did set --force flag)\n')
        else:
            sys.stdout.write('Overwrite MTZ file? Enter "yes" to confirm ')
            if raw_input().lower() != 'yes':
                sys.stdout.write('Phew... that was close...\n')
                sys.exit(1)
            else:
                sys.stdout.write('Well, if you insist...\n')
    sys.stdout.write('Writing the output MTZ file '+args.mtzout + '... ')
    mtz.write(args.mtzout)
    sys.stdout.write('done\n')
