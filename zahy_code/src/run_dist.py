from test import *
from sys import argv

def run_write_dist(inst_ind, astar):
    inst=instance_set[inst_ind]
    hist=instance_distribution(inst,astar,sample_size=100,include_location=True)
    write_distribuition_raw(inst,astar)

#astar_solvers=[make_Astar(heur=zeroh),make_Astar(lapse_rate=.01),make_Astar(lapse_rate=.1),make_Astar(calcF=make_fCalc(0.9,1,1),heur=randh)]

astar_solvers=[make_Astar(lapse_rate=.1),make_Astar(calcF=make_fCalc(0.9,1,1),heur=randh)]

try:
	inst_ind = int(argv[1])
except:
	print 'ERROR: {} instance_index'.format(argv[0])
for astar in astar_solvers:
	run_write_dist(inst_ind,astar)

