import os
from glob import glob
import pandas
import argparse
import dateutil
import numpy as np

"""
get all spreadsheets

grab first one

set up dict mapping subid -> baseline date
setup all dframes(one per test)
"""

def xls_to_df(infile):
    """Uses Pandas to turn an excel file into a dataframe
    Assumes you want to parse Sheet1"""
    try:
	df = pandas.ExcelFile(infile).parse('Sheet1')
    	return df
    except:
	df = pandas.ExcelFile(infile)
        raise IOError('unable to parse Sheet1, sheets:%s'%df.sheet_names)
    

def sub_baseline_dict(spreadsheet, id_header, baseline_date_hdr):
    df = xls_to_df(spreadsheet)

    baseline_dates = {}
    subj_id = df[id_header].values
    #date = df[baseline_date_hdr].values
    for id in subj_id:
	try:
 	    datestr = df[df[id_header] == id][baseline_date_hdr].item()
	    #date = dateutil.parser.parse(datestr)
            baseline_dates.update({id: datestr})
	except:
	    raise ValueError('unable to parse date, %s:%s'%(id,datestr))
    return baseline_dates 
    

    """ make a dictionary mapping subid to baseline date"""
    pass

def count_sessions(spreadsheets,valid_cols):
    """Return a dataframe with the number of datapoints found for each subject for each test."""
    possible_sessions = len(spreadsheets)
    df0 = xls_to_df(spreadsheets[0])
    nsub, _ = df0.shape
    ntests = len(valid_cols)
    counts = np.zeros((nsub,ntests))
    for ss in spreadsheets:
        tempdf = xls_to_df(ss)
	testsdf = tempdf.reindex(columns = valid_cols).values
	hasdata = pandas.notnull(testsdf)
	counts += hasdata
    newdf = pandas.DataFrame(counts, index = df0.index, columns = valid_cols)
    return newdf, possible_sessions

def make_sheets(tests,subs,poss_sess)
    dframes = {}
    for test in tests:
	temp = np.array((len(subs)),poss_sess)
	temp[!] = np.nan
	cols = ['%s_v%'%(test,x) for x in range(poss_sess)]
	othercols = ['s%_v%_delta'%(test,x) for x in range(poss_sess)]
	allcols = cols + othercols
	allcols.sort()
	tmpdf = pandas.DataFrame(tmp,index = subs, columns = allcols)
	dframes.update({test: tmpdf})
    
    return dframes


def setup_dataframes(visitone_sheet, id_header,  n_possible_sessions):
    df = xls_to_df(visitone_sheet)
    subj_ids = df[id_header].values
    empty = np.empty((len(subj_id),6))
    empty[:] = np.nan
    cols = df.columns
    cols = [x for x in cols if 'TstScrs' in x]
    for test in cols:
        newdf = pandas.DataFrame(empty, columns = ['nsess', 'poss_sess', 'slope','mean', 'std', 'baseline'], index = subj_id)
     
	pass    		
    """next step is to manipulate the dataframe within the loop"""
    """create a  dataframe from spreadsheet 1 for each test
    return a list of dataframes"""
    pass




if  __name__ == '__main__':

    spreadsheet_dir = '/home/jagust/bacs_pet/projects/jake/longdat/'
    globstr = 'LongiSubjs_S*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr)))
