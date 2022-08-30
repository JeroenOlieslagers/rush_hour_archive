from sys import argv
from analyze import *

"""

"""
try:
    filename = argv[1]
    index = int(argv[2])
except:
    print 'usage: analyize_real_dist.py <filename> <index>'
calc_real_dist(filename,index)
