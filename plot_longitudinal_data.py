import numpy as np
import pylab
import pandas

def retrieve_vals(panel,test):
    nsess,jnk,_ = panel.shape
    #val0 = panel['sess_00'][:][test].values.copy()
    #val1 = panel['sess_02'][:][test].values.copy()
    #for block in range(3,nsess,1):
       # val = panel['sess_0%d'%(block)][:][test]
       #val1.update(val)
    val0 = panel.minor_xs(test).min(axis=1)
    val1 = panel.minor_xs(test).max(axis=1)
    val0.index = panel['sess_00'][:]['TstScrs ::BAC#'].tolist()
    val1.index = panel['sess_00'][:]['TstScrs ::BAC#'].tolist()
    

    return val0,val1

def plot_test_vals(panel,slopedf,intdf,test):
    nsub,jnk = slopedf.shape
    x0,x1 = retrieve_vals(panel,'days_since_sess1')
    x_alt = panel['sess_01'][:]['days_since_sess1']
    y = panel['sess_00'][:][test]
    y_alt = panel['sess_01'][:][test]
    m = slopedf[:][test]
    b = intdf[:][test]
    arr_list = [x1,y,m,b,x_alt,y_alt]
    val_mat = x0.append(arr_list)
    val_mat = np.reshape(val_mat,(nsub,7),'F')
    valdf = pandas.DataFrame(val_mat)
    pylab.figure()
    for _,x0,x1,y,m,b,x_alt,y_alt in valdf.itertuples():
	if np.isnan(y) == False:
	    xs = [x0,x1]
	    ys = [m * x0 + b, m * x1 + b]
	    if m < 0:
	        pylab.plot(xs,ys, 'r-', lw = 1)
	    else:
		pylab.plot(xs,ys, 'k-', lw = 1)
        else:
	    xs = [x_alt,x1]
	    ys = [x_alt * m + b, m * x1 + b]
	    if m < 0:
	        pylab.plot(xs,ys, 'r-', lw = 1)
	    else:
                pylab.plot(xs,ys, 'k-', lw = 1)

    pylab.show()
    #return valdf
