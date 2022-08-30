from pymc3 import Model, DiscreteUniform, Uniform, Discrete, Deterministic
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from csv import DictReader
from random import random

def make_fCalc(gF=1,hF=1,gAddition=1):
    def tmp_f(g,h,s):
        return (gF*(g+gAddition)+hF*h,g+gAddition,hF,s)
    tmp_f.__name__='{0}*(g+{1})+{2}h'.format(gF,gAddition,hF)
    return tmp_f


def LRTA(start,heur=lambda x:0,calcF=make_fCalc(),is_stop=lambda x:False,update_h=True,hcache={},iters=1):
    return [1 for _ in range(9)],[]



class IVS(Discrete):
    def __init__(self,instances,h_epsilon,learning_iter, *args, **kwargs):
        super(IVS, self).__init__(*args, **kwargs)#What does discrete expects?
        self.instances = instances
        self.h_epsilon = h_epsilon
        self.learning_iter = learning_iter

    def logp(self, value):
        print value
        return logp(self.instances,self.h_epsilon,self.learning_iter,value)

def make_model_path_length(i):
    #@as_op(itypes=[tt.dscalar,tt.lscalar], otypes=[tt.lscalar])
    def model_path_length(h_epsilon,learning_iter):
        path,_= LRTA(i,heur=lambda x: (1+h_epsilon)*len(x),update_h=True,iters=learning_iter)
        print len(path)
        return len(path)
    return model_path_length

# TODO: store per trial 
def logp(instances,h_epsilon,learning_iter,sub_path_length):
    max_trials=20
    lp=0
    trials=1;
    for i,sr in zip(instances,sub_path_length):
        m_path_f = make_model_path_length(i)
        while True:
            r=m_path_f(h_epsilon,learning_iter)
            print 'Model:{} Subject:{}'.format(r,sr)
            if  r ==  sr or trials > max_trials:
                print trials
                break
            trials+=1
        lp+=sum([(1./t) for t in range(1,trials+1)])
        trials=1
    return -lp


lrta_model=Model()
with lrta_model:
    learning_iter=DiscreteUniform('learning_iter',lower=1, upper=5)
    h_epsilon = Uniform('h_epsilon',lower=0.,upper=1.)
    instances=np.array([0])
    path_length=IVS('path_length',instances,h_epsilon,learning_iter, observed=[9])


