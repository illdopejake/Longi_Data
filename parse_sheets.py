import os
from glob import glob
import pandas
import argparse

"""
get all spreadsheets

grab first one

set up dict mapping subid -> baseline date
setup all dframes(one per test)
"""

def xls_to_df(infile):
    pass

def sub_baseline_dict(spreadsheet, id_header, basline_date_hdr):
    """ make a dictionary mapping subid to baseline date"""
    pass

def setup_dataframes(visitone_sheet, n_possible_sessions):
    """ create a  dataframe from spreadsheet 1 for each test
    return a list of dataframes"""
    pass




if  __name__ == '__main__':

    spreadsheet_dir = '/path/to/spreadsheets'
    globstr = 'BACS_neurpsych*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr))
