#! /usr/bin/env python3

# There should be a way to do this with relative imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    headerhelp = \
'''
MTZ2NPY converts an mtz-file to numpy readable npy file.
'''
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=headerhelp)
    parser.add_argument('inpath',
                        help='The input MTZ file.')
    parser.add_argument('outpath',
                        help='The output NPY file.')
    parser.add_argument('-z', '--gzipped', action='store_true', 
                        help='Compress the output.')
    parser.add_argument('-f', '--overwrite', action='store_true', 
                        help='Overwirte existing files.')
    
    args = parser.parse_args()
    
    from pymtz import read_mtz_file
    
    print("Reading ",args.inpath, end='...', flush=True)
    dataset = read_mtz_file(args.inpath)
    print("done")
    print("Writing ",args.outpath, end='...', flush=True)
    dataset.savenpy(args.outpath, fOverwrite=args.overwrite, fCompressed=args.gzipped)
    print("done")
    

if __name__ == "__main__":
    main()
