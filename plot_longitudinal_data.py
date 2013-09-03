import numpy as np
import pylab
import pandas

def retrieve_vals(panel,test):
    nsess,jnk,_ = panel.shape
    val0 = panel['sess_00'][:][test]
    val1 = panel['sess_02'][:][test]
    for block in range(3,nsess,1):
        val = panel['sess_0%d'%(block)][:][test]
	val1.update(val)
    
    val0.index = panel['sess_00'][:]['BAC#'].tolist()
    val1.index = panel['sess_00'][:]['BAC#'].tolist()

    return val0,val1

def plot_test_vals(panel,slopedf,intdf,test):
    nsub,jnk = slopedf.shape
    x0,x1 = retrieve_vals(panel,'days_since_sess1')
    x_alt = panel['sess_01']
    y = panel['sess_00'][:][test]
    m = slopedf[:][test]
    b = intdf[:][test]
    arr_list = [x1,y,m,b]
    val_mat = x0.append(arr_list)
    val_mat = np.reshape(val_mat,(nsub,5),'F')
    valdf = pandas.DataFrame(val_mat)
    pylab.figure()
    for _,x0,x1,y,m,b in valdf.itertuples():
        x = [x0,x1]
	y = [y, m * y + b]
	pylab.plot(x,y, 'k-', lw = 1)
    
    pylab.show()
