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
    for sn, ss in enumerate(spreadsheets):
	vcols = [x.replace('Session 1', 'Session %d'%(sn+1)) for x in valid_cols]
        tempdf = xls_to_df(ss)
	testsdf = tempdf.reindex(columns = vcols).values
	hasdata = pandas.notnull(testsdf)
	counts += hasdata
    newdf = pandas.DataFrame(counts, index = df0.index, columns = valid_cols)
    return newdf, possible_sessions

def make_sheets(tests,subs,poss_sess):
    """ creates empty dataframe to hold score and time delta values
    for each test, for len(subs)"""
    dframes = {}
    for test in tests:
	temp = np.empty((len(subs),(2 * poss_sess)))
	temp[:] = np.nan
	cols = ['%s_%d'%(test,x) for x in range(poss_sess)]
	othercols = ['%s_%d'%(test,x)+'_delta' for x in range(poss_sess)]
	allcols = cols + othercols
	allcols.sort()
	tmpdf = pandas.DataFrame(temp,index = subs, columns = allcols)
	dframes.update({test: tmpdf})
    
    return dframes

def populate_test_dfs(dframes, spreadsheets, baseline_dates):
    """ populate data frames with test scores and delta time since
    baseline visit

    dframes : dict
        empty data frame for each test to hold test score and time delta
	(to be filled)
    spreadsheets : list
        list of strings pointing to xls files holding test data 
	(in visit order)
    baseline_dates: dict
        maps subid -> baseline test date
    """
    used_tests = sorted(dframes.keys())
    for sn, sheet in enumerate(spreadsheets):
        tmpdf = xls_to_df(sheet)
	vcols = [x.replace('Session 1', 
		           'Session %d'%(sn+1)) for x in used_tests]
	

def create_rfx_dataframe(spreadsheets, valid_tests = None):
    """Given spreadsheets with tests, create a numpy matrix
    with scores, delta time (from baseline) in days, and 
    possible number tests for each subject for each test
    in valid_tests (if valid_tests is None, use all of them)
    to access
    panel.items (0)-> sessions
    panel.major_axis (1)-> subjects
    panel.minor_axis (2)-> tests"""
    alldf = {}
    for sn, sheet in enumerate(spreadsheets):
        tmpdf = xls_to_df(sheet)
        newtmpdf = fix_column_names(tmpdf, sn+1)
        alldf.update({'sess_%02d'%sn : newtmpdf})
        newpanel = pandas.Panel(alldf)
    newcols = newpanel.minor_axis.tolist() + [unicode('days_since_sess1')]
    newpanel = newpanel.reindex(minor_axis=newcols)

    return newpanel


def fix_column_names(dataframe, session_number):
    """ removed reference to test session number in column names"""
    cols = dataframe.columns
    newcols = [x.replace('Session %d'%session_number, '') for x in cols]
    newcols = [x.replace('Test Date %d'%session_number, 
                         'Test Date') for x in newcols]
    dataframe.columns = newcols
    return dataframe

def add_deltadays_topanel(panel):
    """ given panel, update delta days"""
    minor_xs = 'TstScrs ::Neuropsych Exam Test Date'
    nsess, nsub, ntests = panel.shape
    ### need to fix this to handle 
    for i in range(nsess):
        panel['sess_%02d'%i][:]['days_since_sess1'] = panel['sess_%02d'%i][:][minor_xs] - panel['sess_00'][:][minor_xs]
    

def setup_dataframes_0(spreadsheets, baseline_dates, tests):
	poss_sess = len(spreadsheets)
	df0 = xls_to_df(spreadsheets[0])
	nsub,_ = df0.shape
	subs = df0.index
	dframes = make_sheets(tests,subs,poss_sess)


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
    globstr = 'Longi_Neuropysch_S*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr)))
    panel = create_rfx_dataframe(spreadsheets)
    add_deltadays_topanel(panel)

