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
    df = pandas.ExcelFile(infile).parse('Sheet1')
    print df
    ###not sure how to make this return a df with a name unique to infile###
    pass

def sub_baseline_dict(spreadsheet, id_header, basline_date_hdr):
    """ make a dictionary mapping subid to baseline date"""
    pass

def setup_dataframes(visitone_sheet, n_possible_sessions):
    """ create a  dataframe from spreadsheet 1 for each test
    return a list of dataframes"""
    pass




if  __name__ == '__main__':

    spreadsheet_dir = '/home/jagust/bacs_pet/projects/jake/longdat/'
    globstr = 'LongiSubjs_S*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr))
