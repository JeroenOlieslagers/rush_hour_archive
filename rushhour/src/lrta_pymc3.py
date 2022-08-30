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
from logp import calc_logp_path_length,get_instance_names_by_subject, get_paths_by_subject

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
    def __init__(self,learning_iter,exp_select, *args, **kwargs):
        super(IBS, self).__init__(*args, **kwargs)#What does discrete expects?
        self.learning_iter = learning_iter
        self.exp = exp_select

    def logp(self, value):
        return calc_logp_path_length_op(self.exp,value)

@as_op(itypes=[tt.dscalar,tt.lvector], otypes=[tt.dscalar])
def calc_logp_path_length_op(exp,value):
    return calc_logp_path_length(RTA(exp=exp),instances,value)


def make_model_path_length(i):
    def model_path_length(learning_iter,exp):
        #path,_= LRTA(i,heur=min_manhattan_distance,update_h=True,iters=learning_iter,exp=exp)
        #return len(path)
        return randint(1,100)
    return model_path_length

@as_op(itypes=[tt.lscalar,tt.dscalar,tt.lvector], otypes=[tt.dscalar])
def logp(learning_iter,exp,sub_path_length):
    k=100
    lp=sum([calc_logp_path_length(learning_iter,exp,sub_path_length) for _ in range(k)])/k
    print 'lp={0}'.format(lp)
    return np.array(-lp)

#mlp should be -0.69
#def calc_logp_path_length(model,sub_path_lengths):
#    lp=0
#    trials=1;
#    for i,sr in zip(instances,sub_path_length):
#        m_path_f = path=make_model_path_length(i)
#        while True:
#            r=m_path_f(learning_iter,exp)
#            if  r ==  sr or trials > max_trials:
#                break
#            trials+=1
#        #print '{0},{1},{2},{3},{4}'.format(subject,i.name,learning_iter,exp,trials)
#        sample_data[(i.name,str(learning_iter))]=trials
#        mlp=sum([(1./t) for t in range(1,trials+1)])
#        lp+=mlp
#        print '  Guessed {0} with {1} trials. mlp={2} lp={3}'.format(sr,trials,mlp,lp)
#        trials=1
#    return lp
#
#

lrta_model=Model()
with lrta_model as lrta_model:
    max_trials=40
    instance_names=get_instance_names_by_subject(path_file,subject)
    instances=np.array([i for i in instance_set if i.name in instance_names]) #is this sorted properly?
    d=get_paths_by_subject(path_file,instance_names,subject,lambda x:int(x['human_length']))
    print 'guessing {}'.format(d)
    sample_data={}
    learning_iter=DiscreteUniform('learning_iter',lower=1, upper=5)
    exp_select = Uniform('exp', lower=4., upper=20.)
    path_length=IBS('path_length',learning_iter,exp_select, observed=d)
    step2=Metropolis(vars=[learning_iter])
    step3=Metropolis(vars=[exp_select])
    trace = sample(2000,step=[step2,step3])
    print '****** loo ********'
    print loo(trace,pointwise=True)


