import os
import unittest 
import numpy.testing as npt
import pandas
import tempfile
import parse_sheets

class TestParseSheets(unittest.TestCase):
    def setUp(self):
        tmpdf = pandas.DataFrame([[1],[2],[3]])
	tempdir = tempfile.mkdtemp()
	outf = os.path.join(tempdir, 'tmpdf.xls')
	tmpdf.to_excel(outf)
	outf2 = os.path.join(tempdir, 'goodtmpdf.xls')
	tmpdf.to_excel(outf2, sheet_name = 'Sheet1')

	self.xls = outf
	self.goodxls = outf2
	self.df = tmpdf

    def test_xls_to_df(self):
	npt.assert_raises(IOError,parse_sheets.xls_to_df,self.xls)
	npt.assert_raises(IOError, 
			  parse_sheets.xls_to_df, 'missing_file.xls')

	good_df = parse_sheets.xls_to_df(self.goodxls)
	npt.assert_equal(good_df.ix[0][0], 1)

    def CleanUp(self):
        pth, _ = os.path.split(self.xls)
	cmd = 'rm -rf %s'%pth
	os.system(cmd)
