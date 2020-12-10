import math
import numpy as np
import scipy as sp


def xytrapz(x, y):
    from scipy.integrate import trapz
    return trapz(y,x)


#------- Save and read data with pickle -------#

def save( file, varnames, values):
    """
    Usage: save('mydata.pysav', ['a','b','c'], [a,b,c] )
    """
    import pickle
    f = open(file,"wb")
    super_var =dict( zip(varnames,values) )
    pickle.dump( super_var, f )
    f.close

def restore(file):
    """
    Read data saved with save function.
    Usage: data = restore('mydata.pysav')
    a = data['a']
    b = data['b']
    c = data['c']
    """
    import pickle
    f = open(file,"rb")
    result = pickle.load(f, encoding='latin1')
    f.close
    return result

#------- Read ascii tables --------#
# June 11, 2013
# needed for computers that don't have access to asciidata (hotfoot)

def read_table( filename, ncols, ignore='#' ):
	"""
	Read data saved in an ascii table
	Assumes data is separated by white space
	Assumes all the data are floats
	Ignores lines that start with the ignore character / sequence
	---------------
	Usage : read_table( filename, ncols, ignore='#' )
	Returns : A dictionary with the column index as keys and the column data as lists
	"""

	# Initialize result dictionary
	result = {}
	for i in range(ncols):
		result[i] = []
	
	try : f = open( filename, 'r' )
	except:
		print('ERROR: file not found')
		return
	
	end_of_file = False
	while not end_of_file:
		try:
			temp = f.readline()
			if temp[0] == ignore : pass  # Ignore the ignore character
			else:
				temp = temp.split()
				for i in range(ncols) : result[i].append( np.float(temp[i]) )
		except:
			end_of_file = True
	
	f.close()
	return result