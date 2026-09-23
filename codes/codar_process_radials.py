"""
Process CODAR RUV files
"""
# Standard modules

import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Scientifc packages

import numpy as np

# Set up the path to find local modules

homepath = os.sep.join(str(Path(__file__).parent).split(os.sep)[:-1])
sys.path.append(homepath)

from hfradar import read_ruv

# Extract the date from the filename.
# CODAR: RDLm_CREU_2025_01_31_1700.ruv


def codar_filename_to_date(filelist, default_date='1970-01-01T00:00'):
    datelist = []
    for file in filelist:
        res = os.path.basename(file).split('.')[0].split('_')
        try:
            day = res[4]
            month = res[3]
            year = res[2]
            hour = res[5][:2]
            minute = res[5][2:]
            file_date = np.datetime64('-'.join([year, month, day]) + 'T' + ':'.join([hour, minute]))
        except (IndexError, ValueError):
            file_date = np.datetime64(default_date)
        datelist.append(file_date)
    return np.array(datelist)


# Search recursively all files with a certain extensions


def _search_files(main_path, ext='ruv'):
    root = Path(main_path)
    return sorted(list(root.rglob(f"*.{ext.lstrip('.')}")))

# Write mesages


def _message(*argv, verbose=True, indent=False, indentchars=4*' '):
    if not verbose:
        return
    out = " ".join([str(a) for a in argv])
    if indent:
        indt = indentchars
    else:
        indt = ''
    print(indt + out)

# Main code


if __name__ == '__main__':

    # Create the parser

    parser = argparse.ArgumentParser(
        prog='codar_process_radials',
        description='Process CODAR .ruv files',
        epilog='')

    parser.add_argument('-p', '--path', default='./')  # Input path
    parser.add_argument('-o', '--output', default='./')  # Output path
    parser.add_argument('-f', '--file', default=None)  # Input file
    parser.add_argument('-c', '-config', default=None)  # Configuration file
    parser.add_argument('-t0', '--date-start', default=None)  # Start date in ISO 8601 format
    parser.add_argument('-tf', '--date-end', default=None)  # End date in ISO 8601 format
    parser.add_argument('-v', '--verbose', default=True)  # Turn on/off verbosity

    # Parse options

    args = parser.parse_args()

    # Turn on/off verbosity

    verbose = args.verbose

    # Search files, if input file or folders exist

    if args.file is None:
        if not os.path.exists(args.path):
            _message('Folder', args.path, 'not found', verbose=verbose)
            exit(1)
        _message("Searching in", args.path, verbose=verbose)
        files_ruv = _search_files(args.path, ext='ruv')
        _message("Found", len(files_ruv), "files", indent=True, verbose=verbose)
    else:
        files_ruv = [args.path + os.sep + args.file]
        if not os.path.exists(files_ruv[0]):
            _message('File', files_ruv[0], 'not found, exiting', verbose=verbose)
            exit()
    files_ruv = np.array(files_ruv)

    # Select the functions that will be used to process files

    filename_to_date = codar_filename_to_date  # Input filenames are RUV files from CODAR

    # Process each individual file

    dates_ruv = filename_to_date(files_ruv)

    # Files to process

    ifiles = [0, 1, 2, 3, 5]

    # Process files

    nfiles = len(ifiles)
    _message("Analysing", nfiles, "files:", verbose=verbose)
    for i in ifiles:
        _message(i+1, "of", nfiles, os.path.basename(files_ruv[i]), dates_ruv[i], verbose=verbose)

        #res = read_ruv(files_ruv[i])


