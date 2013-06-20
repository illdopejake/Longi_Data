import os
from glob import glob
import pandas
import argparse
import dateutil

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

def setup_dataframes(visitone_sheet, id_header,  n_possible_sessions):
    df = xls_to_df(visitone_sheet)
    subj_id = df[id_header].values
    empty = np.empty((len(subj_id),6))
    for test in df.iteritems():
        newdf = pandas.DataFrame(empty, columns = ['nsess', 'poss_sess', 'slope','mean', 'std', 'baseline'], index = subj_id	    
	pass    		
    """create a  dataframe from spreadsheet 1 for each test
    return a list of dataframes"""
    pass




if  __name__ == '__main__':

    spreadsheet_dir = '/home/jagust/bacs_pet/projects/jake/longdat/'
    globstr = 'LongiSubjs_S*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr)))
