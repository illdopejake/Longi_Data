import os
from glob import glob
import pandas
import argparse
import dateutil
import datetime
import numpy as np
import collections

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

def count_sessions(spreadsheets):
    """Return a dataframe with the number of datapoints found for each subject for each test."""
    possible_sessions = len(spreadsheets)
    df0 = xls_to_df(spreadsheets[0])
    newcols = df0.columns.tolist()
    newcols = [x for x in newcols if 'Exam Test Date' not in x]
    valid_cols = [x for x in newcols if 'Session Notes' not in x]
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
    newdf = newdf.sort(axis = 1)
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
    newcols = sorted(newpanel.minor_axis.tolist() + [unicode('days_since_sess1')])
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
    
def calculate_longitudinal_scores(panel,fpanel):
    #fpanel = create_masked_xdf_panel(panel)
    nsess, nsub, ntests = fpanel.shape
    #calculate denominator
    #xdf = fpanel.minor_xs(test)
    #EX = np.sum(xdf, axis = 1)
   # EX = xdf.sum(axis = 1)
    #D_coef_1 = EX**2
    ncountdf,_ = count_sessions(spreadsheets)
    #ncountdf = ncountdf.drop('BAC#',1)
    #newcol = np.zeros(nsub)
    #newcol[:] = np.nan
    #ncountdf['TstScrs ::CVLT Total Recs'] = newcol
    recols = sorted(ncountdf.columns.tolist())
    vcols = [x.replace('Session 1','') for x in recols]
    ncountdf.columns = vcols

    #make sure to change this last part ^^^ before rerunning script with new data
    
    EX_df = fpanel.sum(axis = 'items')
    D_coef_1 = EX_df**2

    exp_panel = fpanel**2
    #D_coef_0 = nsess * (exp_panel.sum(axis = 'items'))
    expsum = exp_panel.sum(axis = 'items')
    ncountdf.index = D_coef_1.index.tolist()
    D_coef_0 = expsum.mul(ncountdf)
    denom = D_coef_0 - D_coef_1
    
    
    #calculate numerator
    
    #ycols = [x for x in npanel.minor_axis if 'days_since' not in x]
    #ycols = [x for x in ycols if 'Session Notes' not in x]
    #ycols = [x for x in ycols if 'BAC#' not in x]
    #ypanel = npanel.reindex(minor_axis=ycols)
    ypanel = adjust_headers(panel)
    ypanel.major_axis = panel['sess_00'][:]['TstScrs ::BAC#']
    _,jnk,Nntest = ypanel.shape
    ydf = ypanel.sum(axis='items')
    #xdf_panel = create_masked_xdf_panel(panel)
    xdf_panel = fpanel
    xdf = xdf_panel.sum(axis = 'items')
    N_coef_1 = pandas.DataFrame((ydf.values * xdf.values),index = ydf.index.tolist(), columns  = ydf.columns.tolist())
    #EX_mat = np.hstack(tuple(EX.tolist() * Nntest))
    #EX_mat = np.reshape(EX_mat, (nsub,Nntest), 'F')
    #N_coef_1 = ydf.mul(EX_mat)
    #newmat = xdf.values.repeat(Nntest, axis = 1)
    #newmat = np.reshape(newmat, (nsub,nsess,Nntest), 'C')
    #xpanel = pandas.Panel(newmat).transpose(1,0,2)
    # I have since realized I can reindex xpanel so the columns match those of 
    #y panel, which will save the necessity of having to use .values.  Add in
    #soon. code: xpanel = pandas.Panel(newmat, major_axis = panel.minor_axis)
    #cmat = xpanel.values * ypanel.values
    #cpanel = pandas.Panel(cmat)
    cpanel = pandas.Panel((xdf_panel.values * ypanel.values), items = panel.items.tolist(),major_axis = ypanel.major_axis.tolist(),minor_axis = ypanel.minor_axis.tolist())
    #N_coef_0 = (cpanel.sum(axis = 'items')) * nsess
    cpansum = cpanel.sum(axis = 'items')
    nncountdf = ncountdf
    nncountdf.index = cpansum.index.tolist()
    N_coef_0 = cpansum.mul(nncountdf)
    numer = N_coef_0.sub(N_coef_1)
    #numer.index = denom.index.tolist()
    #numer.columns = denom.columns.tolist()
    #prenumr = N_coef_0.values - N_coef_1.values
    #numer = pandas.DataFrame(prenumr)

    #Prepare and run equation
    slopedf = numer.div(denom, axis = 'index')
    slopedf = slopedf.drop('TstScrs ::BAC#',1)
    #meandf = ypanel.mean(axis = 'items')
    
     
    #use mask generated by output slope df to create new panel masking out subjects for whom the slope has already been calculated. Add time_since_sess1.
    panels = [ypanel,xdf_panel]
    next_panels = []
    rand_test = 'TstScrs ::VR Recognition Total'
    na_generator = xdf_panel.minor_xs(rand_test)
    na_array = np.sum(na_generator,axis = 1)
    na_df = np.hstack(tuple(na_array.tolist() * ntests))
    na_df = np.reshape(na_df, (nsub,ntests), 'F')
    for pan in panels:
        negmask = np.isnan(na_df)
        masked_values = np.where(negmask,pan.values,np.nan)
	masked_panel = pandas.Panel(masked_values,items = ypanel.items.tolist(), major_axis = ypanel.major_axis.tolist(), minor_axis = ypanel.minor_axis.tolist())
        next_panel = masked_panel.drop(masked_panel.items[-1], axis = 0)
	next_panels.append(next_panel)
	#newcols = ypanel.minor_axis.tolist() + [unicode('days_since_sess1')]
	#next_panel.minor_axis = newcols
	#next_panels.append(next_panel)
    #xdf_cols = xdf.columns.tolist()
    #delcols = [x for x in xdf_cols if x != xdf_cols[-1]]
    #nxdf = xdf.reindex(columns = delcols)
    #nxdf = nxdf.transpose()
    #xdf_vals = nxdf.values
    #msess,msub,mtest = next_panel.shape
    #pan_vals = next_panel.values
    #r_pan_vals = np.reshape(pan_vals,(msess,(msub*mtest)), 'F')
    #nextmat = np.concatenate((r_pan_vals,xdf_vals), axis = 1)
    #nextmat = np.reshape(nextmat,(msess,msub,(mtest+1)), 'F')
    #next_panel = pandas.Panel(nextmat, items = next_panel.items.tolist(), major_axis = next_panel.major_axis.tolist())
    #newcols = ypanel.minor_axis.tolist() + [unicode('days_since_sess1')]
    #next_panel.minor_axis = newcols 
    

    return slopedf,next_panels

def adjust_headers(panel):
    ycols = [x for x in panel.minor_axis if 'days_since' not in x]
    ycols = [x for x in ycols if 'Session Notes' not in x]
    ycols = [x for x in ycols if 'TstScrs ::BAC#' not in x]
    ycols = [x for x in ycols if 'Exam Test Date' not in x]
    ypanel = panel.reindex(minor_axis=ycols)

    return ypanel


def generate_others_dfs(panel):
    ypanel = adjust_headers(panel)
    meandf = ypanel.mean(axis = 'items')
    stddf = ypanel.std(axis = 'items')
    maxdf = ypanel.max(axis = 'items')
    mindf = ypanel.min(axis = 'items')
    #baseline_df = panel['sess_00'][:][:]
    sess00 = ypanel['sess_00'][:][:]
    sess01 = ypanel['sess_01'][:][:]
    bs_mask = ~pandas.notnull(sess00.values)
    nbaseline_df = np.where(bs_mask,sess01.values,sess00.values)
    baseline_df = pandas.DataFrame(nbaseline_df,columns = ypanel.minor_axis.tolist())

    return meandf,stddf,maxdf,mindf,baseline_df


def generate_all_slope_dfs(panel):
    nsess,nsub,ntest = panel.shape
    dfs = {}
    fpanel = create_masked_xdf_panel(panel)
    df, nextpanels = calculate_longitudinal_scores(panel,fpanel)
    dfs.update({nsess: df})
    for block in range((nsess-1),2,-1):
        df, nextpanels = calculate_longitudinal_scores(nextpanels[0],nextpanels[1])
	df.index = panel['sess_00'][:]['TstScrs ::BAC#'].tolist()
	dfs.update({block: df})
        
    
    return dfs

def combine_dfs(panel,dfs):
    nsess,_,jnk = panel.shape
    ypanel = adjust_headers(panel)
    xcols = panel['sess_00'][:]['TstScrs :: BAC#'].tolist()
    ycols = ypanel.minor_axis.tolist()
    final_df = dfs[3]
    for block in range(4,(nsess+1)):
        final_df = final_df.combine_first(dfs[block])
    
    #final_df.index = xcols 
    #final_df.columns = ycols 
    
    return final_df


def calculate_intercept(panel,slopedf):
    #(EY-b(EX)/n   b = slope:
    fpanel = create_masked_xdf_panel(panel)
    nsess,_,jnk = panel.shape
    nsub,ntest = slopedf.shape
    ypanel = adjust_headers(panel)
    coef0 = ypanel.sum(axis = 'items')
    coef0.index = slopedf.index.tolist()
    #newcols = [x for x in panel.minor_axis if 'Exam Test Date' not in x]
    #npanel = panel.reindex(minor_axis=newcols)
    #xdf = npanel.minor_xs('days_since_sess1')
    #EX = xdf.sum(1)
    #EX_mat = np.hstack(tuple(EX.tolist() * ntest))
    #EX_mat = np.reshape(EX_mat,(nsub,ntest),'F')
    xdf = fpanel.sum(axis = 'items')
    coef1 = slopedf.mul(xdf)
    #coef1 = pandas.DataFrame((EX_mat * slopedf.values),index = slopedf.index.tolist(), columns = slopedf.columns.tolist())
    numer = coef0.sub(coef1)
    
    ncountdf,_ = count_sessions(spreadsheets)
    #ncountdf = ncountdf.drop('TstScrs ::BAC#',1)
    #newcol = np.zeros(nsub)
    #newcol[:] = np.nan
    #ncountdf['TstScrs ::CVLT Total Recs'] = newcol
    #recols = sorted(ncountdf.columns.tolist())
    vcols = [x.replace('Session 1','') for x in ncountdf.columns.tolist()]
    ncountdf.columns = vcols
    ncountdf.index = numer.index.tolist()
    
    
    intdf = numer.div(nsess)
    #intdf = intdf.drop('TstScrs ::BAC#',1)
    return intdf

    #add new ind and cols to EX_mat, then multiply and finish equationnd.
    #this returns mostly NAN (why not coef0?).  I will have to iterate through like I did for the slope and recursively replace the NANs with values.  I can do this either now or once the equation has been finished.    

def remove_subjs_w_less_than_3_sessions(panel,countdf,df):
    countdf = countdf.drop('TstScrs Session 1::BAC#',1)
    cols = sorted(countdf.columns.tolist())
    countdf.columns = cols
    countmask = np.where(countdf > 2,countdf,np.nan)
    #cdf = pandas.DataFrame(countmask,index = panel['sess_00'][:]['BAC#'].tolist(), columns = df.columns.tolist())
    newmask = np.isnan(countmask)
    maskvals = np.where(newmask,np.nan,df)
    maskdf = pandas.DataFrame(maskvals, index = df.index.tolist(),columns = df.columns.tolist())
    
    return maskdf


def create_masked_xdf_panel(panel):
    nsess,jnk,_ = panel.shape
    pandfs = []
    maskdfs = {}
    xdf = panel.minor_xs('days_since_sess1')
    ypanel = adjust_headers(panel)
    tests = ypanel.minor_axis.tolist()
    for block in range(nsess):
	df = ypanel['sess_0%d'%block]
	pandfs.append(df)

    for i,df in enumerate(pandfs):
	mdf = ~pandas.notnull(df.values)
	maskdfs.update({'sess_0%d'%i: mdf})
    
    nmaskdfs = collections.OrderedDict(sorted(maskdfs.items()))
    maskpanel = pandas.Panel(nmaskdfs,items = panel.items.tolist(),major_axis = panel['sess_00'][:]['TstScrs ::BAC#'].tolist(),minor_axis = tests)
    new_xs = {}
    for test in tests:
	mask = maskpanel.minor_xs(test)
        newdf = np.where(mask,np.nan,xdf.values)
        new_xs.update({test: newdf})
    nnew_xs = collections.OrderedDict(sorted(new_xs.items()))
    fpanel = pandas.Panel(nnew_xs)
    fpanel = fpanel.transpose(2,1,0)
    fpanel.major_axis = panel['sess_00'][:]['TstScrs ::BAC#'].tolist()
    fpanel.items = panel.items.tolist()

    return fpanel

# this works. next, need to somehow apply it to the calc function


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

    #spreadsheet_dir = '/home/jagust/bacs_pet/projects/jake/longdat/old_neuropsych/'
    spreadsheet_dir = '/home/jagust/bacs_pet/projects/jake/longdat/'
    globstr = 'Longi_Neuropysch_S*.xls'
    spreadsheets = sorted(glob(os.path.join(spreadsheet_dir, globstr)))
    panel = create_rfx_dataframe(spreadsheets)
    add_deltadays_topanel(panel)
    #dfs = generate_all_slope_dfs(panel)
    #slopedf = combine_dfs(panel,dfs)
    fpanel = create_masked_xdf_panel(panel)
    slopedf,_ = calculate_longitudinal_scores(panel,fpanel)
    intdf = calculate_intercept(panel,slopedf)
    meandf,stddf,maxdf,mindf,baseline_df = generate_others_dfs(panel)
    #count_df = 
    dflist = [meandf,stddf,maxdf,mindf,baseline_df]
    for df in dflist:
	new_ind = panel['sess_00'][:]['TstScrs ::BAC#'].tolist()
	df.index = new_ind
    countdf,_ = count_sessions(spreadsheets)
    dflist.append(slopedf)
    dflist.append(intdf)
    dfs = []
    for df in dflist:
        df = remove_subjs_w_less_than_3_sessions(panel,countdf,df)
	dfs.append(df)
    
    namelist = ['meandf','stddf','maxdf','mindf','baseline_df','slopedf','intdf']
    outdir = '/home/jagust/bacs_pet/projects/jake/longdat/ps_script_output/'
    for i,name in enumerate(namelist):
	dfs[i].columns = [x.replace(x,x+'_'+name) for x in dfs[i].columns.tolist()]
        dfs[i].to_excel(outdir+name+'.xls', na_rep='')
   
    #intdf = dfs[6]
    #slopedf = dfs[5]
    #baseline_df = dfs[4]
    #mindf = dfs[3]
    #maxdf = dfs[2]
    #stddf = dfs[1]
    #meandf = dfs[0]

    
    #slopedf.to_excel(outdir+'slopedf.xls')
    #meandf.to_excel(outdir+'meandf.xls')
    #stddf.to_excel(outdir+'stddf.xls')
    #maxdf.to_excel(outdir+'maxdf.xls')
    #mindf.to_excel(outdir+'mindf.xls')
    #baseline_df.to_excel(outdir+'baseline_df.xls')
