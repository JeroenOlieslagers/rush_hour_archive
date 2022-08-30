import logging
from pymc3 import Model, DiscreteUniform, Uniform, Discrete, Deterministic, sample, loo, Metropolis
import theano.tensor as tt
import theano
from theano.compile.ops import as_op
from rushhour import min_manhattan_distance, opt_solution_instances
from astar import LRTA, eval_theano,RTA
import numpy as np
from test import instance_set
from random import random,randint
from sys import argv
from logp import calc_logp_path_length,get_instances_by_subject, get_paths_by_subject

logging.basicConfig(level=logging.DEBUG)
#Set these for debugging
#theano.config.exception_verbosity='high'
#theano.config.optimizer='fast_compile'
# Takes two args
try:
    subject=argv[1]
    path_file=argv[2]
except:
    print 'Args:  <subject> <path_file>'

class IBS(Discrete):
    def __init__(self,exp_select, *args, **kwargs):
        super(IBS, self).__init__(*args, **kwargs)#What does discrete expects?
        self.exp = exp_select

    def logp(self, value):
        return calc_logp_path_length_op(self.exp,value)

class dummy_model:
    def solve(solve,instance):
        return [1 for _ in range(randint(13,14))]

@as_op(itypes=[tt.dscalar,tt.lvector], otypes=[tt.dscalar])
def calc_logp_path_length_op(exp,value):
    """
    the decorator strips the theano variables
    """
    return calc_logp_path_length(RTA(exp=exp),instances,value)
    #return calc_logp_path_length(dummy_model(),instances,value)

rta_model=Model()
with rta_model as rta_model:
    logging.info('starting RTA in model context')
    instances=get_instances_by_subject(path_file,subject,instance_set)
    d=get_paths_by_subject(path_file,subject,lambda x:int(x['human_length']))
    exp_select = Uniform('exp', lower=4., upper=20.)
    path_length=IBS('path_length',exp_select, observed=d)
    step1=Metropolis(vars=[exp_select])
    trace = sample(2,step=[step1])
    print '****** loo ********'
    print loo(trace,pointwise=True)


