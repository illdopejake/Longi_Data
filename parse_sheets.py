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

def sub_baseline_dict(spreadsheet, id_header, baseline_date_hdr):
    xls_to_df(spreadsheet)

    baseline_dates = {}
    subj_id = df[id_header]
    date = df[baseline_date_hdr]
    for id in subj_id:
        baseline_dates.update({subj_id: date}) 
    
    ###This won't work for two reasons.
    ###1) I don't think xls_to_df will actually output df, will it?
    ###2) Using df[x] creats an object that is unhashable, I guess because it
    ###returns not just the values in the columns but also an index.  For 
    ###whatever reason, this seems to give the dict trouble :-/
        

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
