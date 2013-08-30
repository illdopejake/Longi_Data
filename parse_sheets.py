import os
from glob import glob
import pandas
import argparse
import dateutil
import datetime
import numpy as np

"""

accessing panel
panel.items (0)-> sessions (gives list of items headers sess_00, sess_01)
panel.major_axis (1)-> subjects (gives list of subject index (0-x)
## to rename major axis
   panel.major_axis = panel.minor_xs('BAC#').values[:,0]
panel.minor_axis (2)-> tests


df = panel['sess_00']
df.ix['BAC001']


get all spreadsheets

grab first one

set up dict mapping subid -> baseline date
setup all dframes(one per test)

tst1 = [x for x in panel.minor_axis if 'Exam Test Date' not in x]
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
	#time_diff = panel['sess_%02d'%i][:][minor_xs] - \
	#	    panel['sess_00'][:][minor_xs]
        time_0 = panel['sess_00'][:][minor_xs]
	time_x = panel['sess_%02d'%i][:][minor_xs]
	time_del = time_x - time_0
	mask = pandas.notnull(time_del)

	time_del_days = [x.days for x in time_del[mask].values]
	panel['sess_%02d'%i][:]['days_since_sess1'][mask] = time_del_days
	print panel['sess_%02d'%i][:]['days_since_sess1']
        #panel['sess_%02d'%i][:]['days_since_sess1'] = time_diff.days
    
def calculate_longitudinal_scores(panel):
    nsess, nsub, ntests = panel.shape
    #calculate denominator
    newcols = [x for x in panel.minor_axis if 'Exam Test Date' not in x]
    npanel = panel.reindex(minor_axis=newcols)
    xdf = npanel.minor_xs('days_since_sess1')
    EX = np.sum(xdf, axis = 1)
    D_coef_1 = EX**2
    D_coef_0 = nsess * (np.sum((xdf**2), axis = 1))
    denom = D_coef_0 - D_coef_1
    
    #calculate numerator
    ycols = [x for x in npanel.minor_axis if 'days_since' not in x]
    ycols = [x for x in ycols if 'Session Notes' not in x]
    ycols = [x for x in ycols if 'BAC#' not in x]
    ypanel = npanel.reindex(minor_axis=ycols)
    _,jnk,Nntest = ypanel.shape
    ydf = ypanel.sum(axis='items')
    EX_mat = np.hstack(tuple(EX.tolist() * Nntest))
    EX_mat = np.reshape(EX_mat, (nsub,Nntest), 'F')
    N_coef_1 = ydf.mul(EX_mat)
    newmat = xdf.values.repeat(Nntest, axis = 1)
    newmat = np.reshape(newmat, (nsub,nsess,Nntest), 'C')
    xpanel = pandas.Panel(newmat).transpose(1,0,2)
    # I have since realized I can reindex xpanel so the columns match those of 
    #y panel, which will save the necessity of having to use .values.  Add in
    #soon. code: xpanel = pandas.Panel(newmat, major_axis = panel.minor_axis)
    cmat = xpanel.values * ypanel.values
    cpanel = pandas.Panel(cmat)
    N_coef_0 = (cpanel.sum(axis = 'items')) * nsess
    prenumr = N_coef_0.values - N_coef_1.values
    numer = pandas.DataFrame(prenumr)

    #Prepare and run equation
    slopedf = numer.div(denom, axis = 'index')
    meandf = ypanel.mean(axis = 'items')
    stddf = ypanel.std(axis = 'items')
    maxdf = ypanel.max(axis = 'items')
    mindf = ypanel.min(axis = 'items')
    baseline_df = npanel['sess_00'][:][:]
    #need to figure out how to iterate through dfs in order to replace indices with those of the original panel (i.e. BAC#, test names, etc)
    
    #use mask generated by output slope df to create new panel masking out subjects for whom the slope has already been calculated. Add time_since_sess1.
    negmask = np.isnan(slopedf)
    masked_values = np.where(negmask,ypanel.values,np.nan)
    masked_panel = pandas.Panel(masked_values,items = ypanel.items.tolist(), major_axis = ypanel.major_axis.tolist(), minor_axis = ypanel.minor_axis.tolist())
    next_panel = masked_panel.drop(masked_panel.items[-1], axis = 0) 
    xdf_cols = xdf.columns.tolist()
    delcols = [x for x in xdf_cols if x != xdf_cols[-1]]
    nxdf = xdf.reindex(columns = delcols)
    nxdf = nxdf.transpose()
    xdf_vals = nxdf.values
    msess,msub,mtest = next_panel.shape
    pan_vals = next_panel.values
    r_pan_vals = np.reshape(pan_vals,(msess,(msub*mtest)), 'F')
    nextmat = np.concatenate((r_pan_vals,xdf_vals), axis = 1)
    nextmat = np.reshape(nextmat,(msess,msub,(mtest+1)), 'F')
    next_panel = pandas.Panel(nextmat, items = next_panel.items.tolist(), major_axis = next_panel.major_axis.tolist())
    newcols = ypanel.minor_axis.tolist() + [unicode('days_since_sess1')]
    next_panel.minor_axis = newcols 
    #next_panel = nextpanel.append(nxdf)
    #figure out how to add nxdf to next_panel. you can either create new header,reindex and add after, or just join them)
    return slopedf, next_panel

def generate_all_slope_dfs(panel):
    nsess,nsub,ntest = panel.shape
    dfs = {}
    df, nextpanel = calculate_longitudinal_scores(panel)
    dfs.update({nsess: df})
    for block in range((nsess-1),2,-1):
        df, nextpanel = calculate_longitudinal_scores(nextpanel)
	dfs.update({block: df})
        
    
    return dfs

def combine_dfs(panel,dfs):
    nsess,_,jnk = panel.shape
    final_df = dfs[nsess]
    for df in dfs.itervalues():
        final_df = final_df.combine_first(df)
	    
	#mask = np.isfinite(df)
	#final_df = np.where(mask,df,final_df)
    
    return final_df

        

   

   # for item in ydf.iteritems():
#	n_coef_1 = item[1] * EX
#	df.update(n_coef_1)


    #newpanel = panel.transpose(2,1,0)
    #frames = [{}
    #for item, frame in newpanel.iteritems():
        #f = np.sum(frame,axis = 1)
	#frames.update({item: f})

def setup_dataframes_0(spreadsheets, baseline_dates, tests):
	poss_sess = len(spreadsheets)
	df0 = xls_to_df(spreadsheets[0])
        #'Session %d'%(sn+1)) for x in used_tests]
	
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

