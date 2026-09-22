"""
Process CODAR RUV files
"""
# Standard modules

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Scientifc packages

import numpy as np

# Set up the path to find local modules

homepath = os.sep.join(str(Path(__file__).parent).split(os.sep)[:-1])
sys.path.append(homepath)

import hfradar

if __name__ == '__main__':

    # Create the parser

    parser = argparse.ArgumentParser(
        prog='codar_process_radials',
        description='Process CODAR .ruv files',
        epilog='')

    parser.add_argument('-p', '--path', default=None)           # Output path
    parser.add_argument('--date-start', default=None)           # Start date in ISO 8601 format
    parser.add_argument('--date-end', default=None)             # End date in ISO 8601 format

    # Parse options

    args = parser.parse_args()





