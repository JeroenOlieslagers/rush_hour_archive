from pymc import DiscreteUniform,Uniform, stochastic,deterministic
from rushhour import min_manhattan_distance, opt_solution_instances
from astar import LRTA
import numpy as np
from test import instance_set
from csv import DictReader
from random import random
from sys import argv


# Two parameters 
try:
    subject=argv[1]
    path_file=argv[2]
except:
    print 'Args:  <subject> <path_file>'

__all__=['instances','path_length','learning_iter',
'h_epsilon']

learning_iter=DiscreteUniform('learning_iter', lower=1, upper=2, doc='learning_iter')
h_epsilon = Uniform('h_epsilon',lower=0.,upper=1.)
exp_select=Uniform('exp', lower=4, upper=20, doc='exp_select')

def get_instances_by_subject(path_file,subject):
    with open(path_file,'rb') as f:
        reader=DictReader(f)
        data=[d['instance'] for d in reader if d['subject']==subject and d['complete']=='True']
        return sorted(data)

instances=np.array([i for i in instance_set if i.name in get_instances_by_subject(path_file,subject)])

def model_path_length(i,h_epsilon,learning_iter,exp_select):
    return len(LRTA(i,heur=lambda x: (1+h_epsilon)*min_manhattan_distance(x),update_h=True,iters=learning_iter,exp=exp_select)[0])


def get_paths_by_subject(path_file,subject,fun):
    instance_names=[i.name for i in instances]
    with open(path_file,'rb') as f:
        reader=DictReader(f)
        data=[(d['instance'],fun(d)) for d in reader if d['subject']==subject and d['instance'] in instance_names]
        return [fr for (x,fr) in sorted(data, key=lambda x: instance_names.index(x[0]))]

def sub_path_length(x):
    return float(x['human_length'])

d=get_paths_by_subject(path_file,subject,sub_path_length)
sample_data={}

def logp(instances,h_epsilon,learning_iter,subject_ratio,exp_select):
    max_trials=20
    lp=0
    trials=1;
    for i,sr in zip(instances,subject_ratio):
        while model_path_length(i,h_epsilon,learning_iter,exp_select)!=sr:
            trials+=1
            if trials > max_trials:
                break
        p_line='{0},{1},{2},{3},{4},{5}'.format(subject,i.name,h_epsilon,learning_iter,exp_select,trials)
	print p_line
        with open('lrta_{0}_{1}'.format(subject,path_file.replace('.','_').replace('/','_')),'a') as f:
            f.write(p_line+'\n')
        sample_data[(i.name,str(h_epsilon),str(learning_iter))]=trials
        lp+=sum([(1./t) for t in range(1,trials+1)])
        trials=1
    return -lp


@stochastic(observed=True)
def path_length(value=d,instances=instances,h_epsilon=h_epsilon,learning_iter=learning_iter,exp_select=exp_select):
    return logp(instances,h_epsilon,learning_iter,d,exp_select)


