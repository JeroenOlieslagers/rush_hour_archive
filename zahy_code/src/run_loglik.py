import logging
from test import *
from astar import LRTA, eval_theano,RTA
from logp import calc_logp_path_length,get_instances_by_subject,get_moves_by_subject, calc_logp_unordered_action, calc_logp_unordered_states
from sys import argv
from itertools import product, islice

logging.basicConfig(level=logging.DEBUG)

def test_log_likelihood(model,ins):
    plan=[RHInstance({'r':(3,2,2),'y':(2,0,3),'o':(4,4,2)},{'p':(5,0,3),'g':(3,4,2)},'easy1'),
          RHInstance({'r':(1,2,2),'y':(2,0,3),'o':(4,4,2)},{'p':(5,0,3),'g':(3,4,2)},'easy1'),
          RHInstance({'r':(1,2,2),'y':(2,0,3),'o':(4,4,2)},{'p':(5,0,3),'g':(3,2,2)},'easy1'),
          RHInstance({'r':(1,2,2),'y':(2,0,3),'o':(1,4,2)},{'p':(5,0,3),'g':(3,2,2)},'easy1'),
          RHInstance({'r':(1,2,2),'y':(2,0,3),'o':(1,4,2)},{'p':(5,0,3),'g':(3,4,2)},'easy1'),
          RHInstance({'r':(1,2,2),'y':(2,0,3),'o':(1,4,2)},{'p':(5,3,3),'g':(3,4,2)},'easy1'),
          RHInstance({'r':(4,2,2),'y':(2,0,3),'o':(1,4,2)},{'p':(5,3,3),'g':(3,4,2)},'easy1'),
          ]
    count,ll=log_likelihood(ins,model,plan)
    print '{0},{1},{2}'.format(count,ll,model.__name__)

models=[RTA(heur=min_manhattan_distance,exp=30), LRTA(heur=min_manhattan_distance,exp=3,iters=3), LRTA(heur=min_manhattan_distance,exp=3,iters=1)]
log_functions=[calc_logp_unordered_states,calc_logp_unordered_action]
subjects=['andra.log','weiji.log','luigi.log','will.log']
runs=product(models,log_functions,subjects)

def run_ll(move_file, run_id):
    logging.info('running run_id {0}'.format(run_id))
    model,func,subject=next(islice(runs,run_id,run_id+1))
    subject_paths=get_moves_by_subject(move_file,subject)
    func(model,subject_paths)
try:
	move_file = str(argv[1])
	run_id  = int(argv[2])
except:
	print 'ERROR: {0} <move_file> <run_id>(0-)'.format(argv[0],len(runs))

run_ll(move_file,run_id)

