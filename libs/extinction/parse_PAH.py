import os
import numpy as np

def file_path( name ):
    return os.path.join(os.path.join(os.path.dirname(__file__).rstrip('extinction'), 'tables'), name)

ION_FILE  = file_path('PAHion_30')
NEU_FILE  = file_path('PAHneu_30')

def parse_PAH( option, ignore='#', flag='>', verbose=False ):

    if option == 'ion': filename = ION_FILE
    if option == 'neu': filename = NEU_FILE

    try : f = open( filename, 'r' )
    except:
        print('ERROR: file not found')
        return

    COLS = ['w(micron)', 'Q_ext', 'Q_abs', 'Q_sca', 'g=<cos>' ]
    result = {}

    end_of_file = False
    while not end_of_file:
        try:
            line = f.readline()

            # Ignore the ignore character
            if line[0] == ignore : pass

            # Characters flagged with '>' earn a dictionary entry with grain size
            elif line[0] == flag :
                gsize = np.float( line.split()[1] )
                result[ gsize ] = {}
                
                # Initialize dictionaries with lists
                for i in range( len(COLS) ) :
                    result[gsize][COLS[i]] = []

            # Sort the columns into the correct dictionary
            else:
                row_vals = line.split()
                for i in range( len(COLS) ) :
                    result[ gsize ][ COLS[i] ].append( np.float( row_vals[i] ) )
        except:
            end_of_file = True

    f.close()

    return result