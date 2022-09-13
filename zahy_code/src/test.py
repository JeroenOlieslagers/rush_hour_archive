from rushhour import *
from math import log
from astar import *
from try_pygame import *
from random import random,seed
from itertools import product
from time import time
from collections import deque
import matplotlib.pyplot as plt
from numpy import var
from collections import deque
from fitting import *
from instance_generator import instance_set_from_jsons
import csv as csv
import time




instance_set_easy=[ RHInstance({'r':(2,2,2),'g':(4,5,2)},{'y':(4,0,3),'p':(2,3,3)},'easy0')
, RHInstance({'r':(3,2,2),'y':(2,0,3),'o':(4,4,2)},{'p':(5,0,3),'g':(3,4,2)},'easy1')
, RHInstance({'r':(2,2,2),'g':(4,3,2),'b':(4,5,2)},{'o':(2,4,2),'y':(4,0,3),'p':(3,3,3)},'easy2')
, RHInstance({'r':(0,2,2),'b':(0,3,3),'g':(4,4,2)},{'y':(2,0,3),'p':(5,1,3)},'easy3')
, RHInstance({'r':(0,2,2),'b':(2,5,3),'o':(3,4,2)},{'g':(1,4,2),'y':(3,0,3),'p':(5,3,3)},'easy4')
,RHInstance({'r':(2,2,2),'g':(0,2,2),'o':(1,4,2), 'b':(3,4,3)},{'y':(5,0,3),'p':(0,3,3)},'easy5')
,RHInstance({'r':(0,2,2),'b':(0,3,3),'y':(3,0,3),'g':(3,4,3)},{'o':(2,4,2),'p':(5,1,3)},'easy6')
,RHInstance({'r':(0,2,2),'b':(2,4,2)},{'y':(3,0,3),'g':(5,3,2),'o':(1,4,2),'p':(4,3,3)},'easy7')
,RHInstance({'r':(2,2,2),'o':(4,5,2)},{'y':(4,0,3),'g':(5,0,2),'b':(3,3,3),'p':(5,2,3)},'easy8')
,RHInstance({'r':(2,2,2),'o':(4,4,2)},{'y':(1,0,3),'g':(2,4,2),'b':(5,0,3),'p':(4,0,3)},'easy9')
,RHInstance({'r':(2,2,2),'o':(4,0,2),'y':(4,3,2)},{'g':(5,1,2)},'easy_or')
,RHInstance({'r':(3,2,2),'o':(4,4,2)},{'y':(3,3,3),'p':(5,1,3)},'easy4_ext')]

instance_set = read_instances()

def print_nodes(nodes):
    for n in nodes:
        print('[{}] loc coverage blocking_nodes'.format(n))
        for l,(cov,blocking_nodes) in nodes[n].items():
            print(' {} {} {}'.format(l,cov,blocking_nodes))


def test_mag(i):
    draw(i)
    mag,nodes=constuct_mag(i)
    mag2dot(mag)
    print(mag)
    print_nodes(nodes)

def report_ASTR_instance(ins,astr):
    plan,stat= astr(ins)
    am=make_Astar(heur=min_manhattan_distance)
    step_n=1
    h_weight=astr.__name__[astr.__name__.rindex('+')+1:astr.__name__.index('histop')]
    for p in plan:
        op,st=am(p)
        d=[ins.name,astr.__name__,h_weight,0,len(plan),len(op),step_n]
        d=[str(x) for x in d]
        step_n+=1
        print(','.join(d))




def report_LRTA_instance(ins,lrta):
    plan,hvals= lrta(ins)
    am=make_Astar(heur=min_manhattan_distance)
    step_n=1
    for p,h in zip(plan,hvals):
        op,st=am(p)
        d=[ins.name,lrta.__name__,len(plan),len(op),step_n]
        d=[str(x) for x in d]
        step_n+=1
        print(','.join(d))




def report_instance(ins,am=make_Astar(heur=min_manhattan_distance)):
    path,stat= am(ins)
    d=[ins.name,len(path),stat['generated'],stat['expanded'],stat['close_size'],stat['open_size']]
    d=[str(x) for x in d]
    print(','.join(d))


def test_stop():
    nodes={'y': {(4, 3, 3): (1.0, ['g'])}, 'p': {(2, 2, 3): (1.0, ['r']), (2, 0, 3): (1.0, ['r']), (2, 1, 3): (1.0, ['r'])}, 'r': {(3, 2, 2): (0.5, ['y']), (4, 2, 2): (1.0, ['y']), (0, 2, 2): (0.5, [])}, 'g': {(2, 5, 2): (1.0, ['p']), (1, 5, 2): (1.0, ['p']), (0, 5, 2): (1.0, ['p'])}}
    visits = dict((n,0) for n in nodes)
    mag = []
    i1 = RHInstance({'r':(2,2,2),'g':(4,5,2)},{'y':(4,0,3),'p':(2,3,3)})
    in_plan=set()
    print(stop_dfs(i1,mag,nodes,'y',visits,in_plan))
    print(stop_dfs(i1,mag,nodes,'p',visits,in_plan))
    print(stop_dfs(i1,mag,nodes,'r',visits,in_plan))
    print(stop_dfs(i1,mag,nodes,'g',visits,in_plan))
    visits['r']=2
    print(stop_dfs(i1,mag,nodes,'r',visits,in_plan))

def test_verify_move():
    i1 = RHInstance({'r':(2,2,2),'g':(4,5,2)},{'y':(4,0,3),'p':(2,3,3)})
    draw(i1)
    print('True: {}'.format(verify_move(i1,('r',-2))))
    print('False: {}'.format(verify_move(i1,('r',-3))))
    print('True: {}'.format(verify_move(i1,('y',+1))))
    print('False: {}'.format(verify_move(i1,('y',-1))))
    print('False: {}'.format(verify_move(i1,('y',+4))))
    print('False: {}'.format(verify_move(i1,('r',+1))))
    print('False: {}'.format(verify_move(i1,('r',+2))))
    print('True: {}'.format(verify_move(i1,('g',-1))))
    print('False: {}'.format(verify_move(i1,('g',-2))))
    print('False: {}'.format(verify_move_for_plan(i1,('r',(3,2,2)))))
    print('True: {}'.format(verify_move_for_plan(i1,('r',(1,2,2)))))
    print('True: {}'.format(verify_move_for_plan(i1,('y',(4,1,3)))))
    print('False: {}'.format(verify_move_for_plan(i1,('y',(4,3,3)))))


def magsize_admissible(instance):
    """
    Shortest path on mag
    """
    root=instance.goal_car
    mag,nodes=constuct_mag(instance) # nodes: {'y': {(4, 3, 3): (1.0, ['g'])}
    print([a for a in mag])
    q=deque(root)
    length=1 #move the goal
    visited=set()
    while len(q)>0:
        n=q.pop()
        if n not in mag or n in visited:
            return length
        else:
            length+=1
            q.extendleft(mag[n].keys())
            visited.add(n)
    return length

def magsize(instance):
    mag,nodes=constuct_mag(instance)
    return sum([len(v) for k,v in mag.items()])

def test_astar_tasks_instance(ins,astar):
    path,stats=astar_tasks(ins,astar)
    stats['solution']=len(path)
    try:
        stats['quality']=len(path)/float(opt_solution_instances[ins.name])
    except:
        stats['quality']='unknown'
    stats['instance']=ins.name
    print(stats)
    print(path)
    show(path,['A*+tasks instance:{}'.format(ins.name),''+str(stats)])


def test_astar_instance(ins,astar):
    path,stats=astar(ins)
    stats['solution']=len(path)
    try:
        stats['quality']=len(path)/float(opt_solution_instances[ins.name])
    except:
        stats['quality']='unknown'
    stats['instance']=ins.name
    print(stats)
    for p in path:
        draw(p)
    show(path,texts=['A* instance:{}'.format(ins.name),''+str(stats)])


def instance_distribution(ins,astar,sample_size=100, include_location=False):
    hist=defaultdict(int)
    for sample in range(sample_size):
        path,stats=astar(ins)
        for p in range(len(path)):
            key=path[p]
            if include_location:
                key=(key,p)
            hist[key]+=1
    return hist

def test_instance_same():
    ins1=deepcopy(instance_set[0])
    print('should be T: {}'.format(ins1 == instance_set[0]))
    d={ins1:1}
    print('should be 1: {}'.format(d[ins1]))
    d[instance_set[0]]=2
    print('should be 2: {}'.format(d[ins1]))
    d[ins1]=3
    print('should be 3: {}'.format(d[instance_set[0]]))

def test_astar(play=False, heur=magsize, entry=make_fCalc(), is_stop=lambda x:False):
    for i in range(len(instance_set)):
        path,stats=Astar(instance_set[i],magsize,entry,is_stop)
        stats['solution']=len(path)
        stats['quality']=len(path)/float(opt_solution_instances[i])
        stats['instance']=i
        stats['h']=heur.__name__
        stats['entry']=entry.__name__
        stats['is_stop']=is_stop.__name__
        print(stats)
        if play:
            show(path,['A* instance:'+str(i),str(stats)])

def test_plan_instance(i):
    show(make_path_for_plan_model(instance_set_easy[i],plan(instance_set_easy[i])),[' plan instance: '+str(i)])

def test_plan_tasks_instance(i,is_easy=False):
    instance=((instance_set,instance_set_easy)[is_easy])[i]
    show(make_path_for_plan_model(instance,plan_with_tasks(instance)),[' plan+task instance: '+str(i)])


def test_plan():
    for i in range(len(instance_set)):
        show(make_path_for_plan_model(instance_set[i],plan(instance_set[i]),' plan_construction instance: '+str(i)))

def g_plus_2h(g,h,s):
    return (g+1+2*h,g+1,2*h,s)
def no_g(g,h,s):
    return (h,g,h,s)
def rand_magsize(instance):
    return magsize*random()
def randh(instance):
    return randint(0,20)
def expandLG1000(stats):
    return stats['expanded']>1000
def expandLG2000(stats):
    return stats['expanded']>2000
def expandLG5000(stats):
    return stats['expanded']>5000
def noisy(heur,mean,sd):
    tmph = lambda instance: heur(instance)+gauss(mean,sd)
    tmph.__name__ = 'noisy_{}_({},{})'.format(heur.__name__,mean,sd)
    return tmph


def draw_dist_instance(instance,alg_name,hist):
    print('~~~~~~~instance {} using {}~~~~~~~~~~~~'.format(instance.name,alg_name))
    max_path=max([k[1] for k in hist])
    samples_per_step=dict([(step,sum([v for k,v in hist.items() if k[1]==step])) for step in range(max_path+1)])
    for k,v in sorted(hist.items(),key=lambda x:x[0][1]):
        print('Step {}:{}{}'.format(max_path-k[1],''.join(['*' for _ in range(v)]),float(v)/samples_per_step[k[1]]))
    for k,v in samples_per_step.items():
        print('Step {}:{}{}'.format(max_path-k,''.join(['=' for _ in range(v)]),v))

def write_distribuition_raw(ins,astar,sample_size=100):
    print('sample step state instance alg ')
    for sample in range(sample_size):
        path,stats=astar(ins)
        for p in range(len(path)):
            print('{} {} {} {} {} '.format(sample, p, path[p].__hash__(), ins.name,astar.__name__.replace(' ','_')))



# The way it should be: Generate a few A* instances with a name: 
# For each A* write_distribution_raw 

#astar_solvers=[make_Astar(),make_Astar(entry=no_g,heur=magsize)]
#
#inst=instance_set_easy[1]
#astar=astar_solvers[0]
#draw(inst)
#test_mag(inst)

#for i in range(10):
#    path,_=astar(inst)
#    draw(path[-1])
#
#for astar,inst in product(astar_solvers,instance_set_easy):
#    print(inst.__hash__())
#    path,_=astar(inst)
#    hist=instance_distribution(inst,astar,sample_size=100,include_location=True)
#    draw_dist_instance(inst,astar.__name__,hist)
#    write_distribuition_raw(inst,astar)



def flip_distribution_raw():
    with open('./results/raw_path_dist.csv','r') as f:
        tmp=[]
        last_step=0
        sample=0
        for l in f:
            l_s = l.split()[0]
            if l_s=='step':
                print(l)
                continue
            if l_s == '0':
                for tl in reversed(tmp):
                    tll=tl.split()
                    tll[0]=str(last_step-int(tll[0]))
                    print(' '.join([str(sample)]+tll))
                sample+=1
                tmp=[]
            else:
                last_step=int(l_s)
            tmp.append(l)
        for tl in reversed(tmp):
            tll=tl.split()
            tll[0]=str(last_step-int(tll[0]))
            print(' '.join([str(sample)]+tll))

def test_piece_possible_locations():
    i=RHInstance({'r':(2,2,2),'g':(0,2,2)},{'y':(4,0,3),'p':(2,3,3)},'easy0')
    i1=RHInstance({'r':(2,2,2),'g':(4,2,2)},{'y':(4,0,3),'p':(2,3,3)},'easy0')
    i2=RHInstance({'r':(1,2,2),'g':(3,2,3)},{'y':(4,0,3),'p':(2,3,3)},'easy0')
    print('should be [(2,2,2),(3,2,2),(4,2,2)]:{}'.format(piece_possible_locations(2,2,2,'h',i,include_current_location=True)))
    print('should be [(3,2,2),(4,2,2)]:{}'.format(piece_possible_locations(2,2,2,'h',i,include_current_location=False)))
    print('should be [(0,2,2),(1,2,2)]:{}'.format(piece_possible_locations(2,2,2,'h',i1,include_current_location=False)))
    print('should be [(0,2,2)]:{}'.format(piece_possible_locations(1,2,2,'h',i2,include_current_location=False)))
    i=RHInstance({'r':(2,2,2),'g':(0,2,2)},{'y':(4,0,2),'p':(4,3,3)},'easy0')
    i1=RHInstance({'r':(2,2,2),'g':(4,2,2)},{'y':(4,2,2),'p':(4,0,2)},'easy0')
    print('should be [(4,1,2)]:{}'.format(piece_possible_locations(4,0,2,'v',i,include_current_location=False)))
    print('should be [(4,3,2),(4,4,2)]:{}'.format(piece_possible_locations(4,2,2,'v',i1,include_current_location=False)))

def test_unlbocked_h():
    for i in range(len(instance_set_easy)):
        draw(instance_set_easy[i])
        print(h_unblocked(instance_set_easy[i]))

def test_learn_mags():
    i = RHInstance({'r':(0,2,2),'o':(3,4,3)},{'g':(2,0,3),'b':(5,0,3)})
    mag1,nodes1 = constuct_mag(i)
    mag2dot(mag1, True)
    i = RHInstance({'r':(0,2,2),'o':(0,4,3)},{'g':(2,0,3),'b':(5,0,3)})
    draw(i)
    mag2,nodes2 = constuct_mag(i)
    mag2dot(mag2, True)
    mag3,nodes3=learn_from_mag(mag1,mag2,i)
    mag2dot(mag3, True)
    print(nodes1)
    print(nodes2)
    print(nodes3)

def compare_tasks_to_no_tasks():
    print('*,instance, solution, solution_with_tasks, opt_solution')
    for i in range(len(instance_set)):
        ins = instance_set[i]
        astar=make_Astar()
        path,_=astar(ins)
        patht,_=astar_tasks(ins,astar)
        print('*,{}, {}, {}, {}'.format(i,len(path),len(patht),opt_solution_instances[i]))

def comp(plan1,plan2):
    print('lengths:{}-{}'.format( len(plan1),len(plan2)))
    for a,b in zip(plan1,plan2):
        print('{}--{}'.format(a,b))
        print(a==b)

def count_till_plan(instance,model,plan,max_try=1000):
    for i in range(1,max_try):
        p1,_=model(instance)
        if p1==plan:
            return i
    return float('inf')


def log_likelihood(instance,model,plan=None):
    c=count_till_plan(instance,model,plan)
    if c==float('inf'):
        return c,-float('inf')
    return c,log(sum([1.0/cc for cc in range(2,c)]))


def log_likelihood_old(instance,model,plan=None,sample_size=100):
    stats=dict(((i,plan[i]),0) for i in range(len(plan)))
    for i in range(sample_size):
        g_p,_ = model(instance)
        for g_i in range(len(g_p)):
            key=(g_i,g_p[g_i])
            #print key
            if key in stats:
                stats[key]+=1
    for i in range(len(plan)):
        print('{0}:{1}'.format(i,stats[(i,plan[i])]),)
    print('')
    zeros = sum(-10 for stp in stats.values() if stp == 0)
    return zeros+ sum(log(float(stp)/sample_size) for stp in stats.values() if stp != 0)

def make_summary(stats):
    text= ['The time allocated for the experiment had passed.','Here are some stats:',' ']
    t='Instance {} {}completed with {}/{} moves ({} solution quality) and {} restarts'
    for x in stats:
        try:
            text.append(t.format(x['instance'],('NOT ','')[x['complete']],x['solution_length'],x['optimal'],x['solution_quality'],x['restarts']))
        except:
            pass
    text.append('_______')
    rs=sum([x['restarts'] for x in stats])
    skips=len([1 for x in stats if not x['complete']])
    text.append('In total, you skipped {} instances and restarted {} instances'.format(skips,rs))
    return text



def run_experiment():
    start_time = time()
    exp_time=(60*300)
    stats=[]
    for i in range(len(instance_set)):
        if (time()-start_time)>exp_time:
            txt=make_summary(stats)
            message_screen(txt,[25,25]+[10]*(len(txt)-2))
            break
        ans,stat=play(instance_set[i])
        stats.append(stat)
        if ans:
            break

def compare_heur():
    am=make_Astar(heur=magsize_admissible)
    a0=make_Astar(heur=zeroh)
    for ins in instance_set:
        print(ins.name)
        path,stat= am(ins)
        print('magsize admissible:')
        print(len(path))
        print(stat)
        path,stat=a0(ins)
        print('zero_h:')
        print(len(path))
        print(stat)

def test_path_length_bug():
    solved=RHInstance( {'B': (0, 5, 3), 'r': (4, 2, 2), 'c': (1, 4, 2), 'w': (4, 0, 2)},{'y': (0, 3, 2), 'b': (0, 0, 3), 'M': (5, 3, 3), 'l': (3, 3, 3)},'Jam-1')
    oneStep = RHInstance({'r': (1, 2, 2), 'B': (0, 5, 3), 'c': (1, 4, 2), 'w': (4, 0, 2)},{'y': (0, 3, 2), 'b': (0, 0, 3), 'M': (5, 3, 3), 'l': (3, 3, 3)},'Jam-1')
    am=make_Astar(heur=min_manhattan_distance)
    solve_instance(solved,True)
    print('****')
    solve_instance(oneStep,True)


def test_456():
    for i in range(100):
        report_instance(instance_set[3],make_Astar(heur=min_manhattan_distance,calcF=make_fCalc(0)))
        report_instance(instance_set[4],make_Astar(heur=min_manhattan_distance,calcF=make_fCalc(0)))
        report_instance(instance_set[5],make_Astar(heur=min_manhattan_distance,calcF=make_fCalc(0)))


def report_instance(ins,am=make_Astar(heur=min_manhattan_distance)):
    path,stat= am(ins)
    d=[ins.name,len(path),stat['generated'],stat['expanded'],stat['close_size'],stat['open_size']]
    d=[str(x) for x in d]
    print(','.join(d))




def solve_instance(ins, show_path=False):
    am=make_Astar(heur=min_manhattan_distance)
    print(ins.name)
    path,stat= am(ins)
    print(am)
    print(len(path)-1)
    print(stat)
    if show_path:
        for pi in path:
            draw(pi)
            print(pi)

def compare_heur_instance(ins):
    #am=make_Astar(heur=magsize_admissible)
    am=make_Astar(heur=min_manhattan_distance)
    a0=make_Astar(heur=zeroh)
    print(ins.name)
    path,stat= am(ins)
    print(am)
    print(len(path))
    print(stat)
    path,stat=a0(ins)
    print(a0)
    print(len(path))
    print(stat)

def test_admissible_md():
    for ins in instance_set[:30]:
        print(ins.name)
        print(opt_solution_instances[ins.name],)
        print(min_manhattan_distance(ins))


def test_admissible():
    for ins in instance_set:
        print(ins.name)
        print(opt_solution_instances[ins.name],)
        print(magsize_admissible(ins))


def test_manhatten():
    for ins in instance_set:
        print('should be 0:' + str(manhattan_distance(ins,ins)))
        ins2 = rand_move(ins)
        print('should be 1:' + str(manhattan_distance(ins,ins2)))

def explort_all_terminals():
    print('terms_by_ins={')
    for ins in instance_set[10:30]:
        t=find_terminal_states(ins)
        print('{}:{},'.format(ins.name,t),)
    print('}')

#solve_instance(instance_set[20])

#test_456()

#play(instance_set[20])
#play(instance_set_easy[1])
#explort_all_terminals()
#test_admissible_md()
#compare_heur_instance(instance_set[20])
#test_path_length_bug()
#solve_instance(instance_set[0], True)

#test_astar_instance(instance_set[20],make_Astar(heur=min_manhattan_distance))

#compare_heur_instance(instance_set[20])
#test_admissible()
#message_screen(['this is a sentence']*10,[25,25]+[10]*8)
# for p in instance_set_easy[1].h:
#    print(p+':' + str(piece_possible_moves(instance_set_easy[1],p)))
# for p in instance_set_easy[1].v:
#    print(p+':' + str(piece_possible_moves(instance_set_easy[1],p)))
#
#show([instance_set_easy[1]])
# astar=make_Astar(heur=magsize)
# path,_=astar(inst)
# show(path)
#
#
#for i in range(10000):
#    print zipf_choice([6,7],1)
#test_astar_instance(ins,make_Astar())
#test_astar_tasks_instance(ins,make_Astar())
#
#for i in range(len(instance_set)):
#    ins=instance_set[i]
#    test_astar_instance(ins,make_Astar())
#    test_astar_tasks_instance(ins,make_Astar())
#
#astar=make_Astar(search_lapse=0.0,lapse_rate=0.0,reconstruct_accuracy=53.0)
#for i in range(10):
#    path,_=astar(instance_set_easy[1])
#    show(path)
#show([instance_set_easy[3]])
def test_minimin():
    ins=instance_set[1]
    zero_dist = min_manhattan_distance(ins)
    print('min_dist: {}, real_dist: {}'.format(zero_dist,opt_solution_instances[ins.name]))
    am=make_Astar()
    path,stat=am(ins)
    print('real_dist'+str(len(path)))
    assert(minimin(ins,0)==zero_dist)
    prv_dist=zero_dist
    for d in range(1,6):
        n_dist=minimin(ins,d)
        print('level:{} val:{}'.format(d,n_dist))
        assert(n_dist>=prv_dist)
        prv_dist=n_dist


def test_fitting(data_file):
    def model_creator(h_weight,g_weight,learning_iters,local_learn):
        model=make_Astar(heur=min_manhattan_distance,calcF=make_fCalc(g_weight,h_weight,1))
        return model

    # define likelihood function:
    def r_square_function(instances,model,data):
        model_data=[]
        subject_data=[]
        for i in instances:
            #print i.name
            path,stat=model(i)
            model_data.append(len(path))
            entry=[d['human_length'] for d in data if d['instance']==i.name and d['complete']=='True']
            print('{} {}'.format(entry[0],len(path)))
            subject_data.append(int(entry[0]))
            residuals = [pow(predicted-int(observed),2) for predicted,observed in zip(model_data,subject_data)]
        return sum(residuals)

    # data
    with open(data_file, 'r') as d:
        reader = csv.DictReader(d, delimiter=',')
        data=[r for r in reader if r['subject']=='andra.log']
    instances_data = set([d['instance'] for d in data])
    instances = [i for i in instance_set if i.name in instances_data]

    # giving initial parameters
    h_weight=1
    g_weight=1
    learning_iters=1
    local_learn = 0
    params=[2*h_weight,0*g_weight,learning_iters,local_learn]
    bounds=[(1,2),(0,1),(1,4),(0,1)]


    #fit 
    print(try_f(instances,model_creator,r_square_function, params,bounds, data))
    #print fit(instances,model_creator,r_square_function, params,bounds, data)

def lrta_marginalization():
    h_epsilon = 0.
    learning_iter=1
    ins = instance_set[0]
    exp_select = 3
    seed(0)
    for i in range(500):
        print(str(len(LRTA(ins,heur=lambda x: (1+h_epsilon)*min_manhattan_distance(x),update_h=True,iters=learning_iter,exp=exp_select)[0])))


def test_models_on_jsons():
    instance_set_jsons = instance_set_from_jsons('.')
    am=make_Astar(heur=h_unblocked)
    for i in instance_set_jsons:
        return i
        # path,stat=am(i)
        # print('optimal path:'+str(len(path)))
        # #show(path, aux=None ,texts=['Astar'])
        # lrta=LRTA(heur=min_manhattan_distance_calc,update_h=True,iters=1500,exp=5)
        # lrta_path=lrta.solve(i)
        # print('lrta path:'+str(len(lrta_path)))
        #show(lrta.solve(i),aux=None ,texts=['lrta'])

def test_models_on_instances():
    instance_set_jsons = instance_set
    am=make_Astar(heur=h_unblocked)
    for i in instance_set_jsons:
        path,stat=am(i)
        print('optimal path:'+str(len(path)))
        #show(path, aux=None ,texts=['Astar'])
        lrta=LRTA(heur=min_manhattan_distance_calc,update_h=True,iters=15,exp=50)
        #show(lrta.solve(i),aux=None ,texts=['lrta'])
        lrta_path=lrta.solve(i)
        print('lrta path:'+str(len(lrta_path)))

def test_peduo_max():
    items = [3,4,5,6,7,8,9,10]
    exp = 50
    for i in range(1000):
        print(items[select_pseudo_max(items,exp)])
#test_peduo_max()
#test_models_on_instances()

# astar=make_Astar(heur=magsize)
inst = test_models_on_jsons()
draw(inst)
#print(magsize_admissible(inst))

do_move(inst, ("1", -2))
do_move(inst, ("7", 3))
do_move(inst, ("6", -1))
do_move(inst, ("5", 2))
do_move(inst, ("4", -3))
do_move(inst, ("3", 3))
do_move(inst, ("0", 2))
do_move(inst, ("r", 2))
do_move(inst, ("2", 3))
do_move(inst, ("r", -1))
do_move(inst, ("4", 1))
do_move(inst, ("5", -4))
# do_move(inst, ("1", -2))
# do_move(inst, ("6", -2))
# do_move(inst, ("4", -1))

draw(inst)
test_mag(inst)

#test_mag(inst)
# am=make_Astar()#heur=magsize)
# start = time.time()
# path, stat = am(inst)
# print(time.time() - start)
# print(len(path), stat)
# show(path)


# draw(inst)
# test_mag(inst)

#lrta_marginalization()

#test_fitting('../results/pilot/paths.csv')

#test_minimin()
#show(RTA(instance_set[0],heur=lambda x: minimin(x,6,h_unblocked)),['RTA'])
#show(RTA(instance_set[1],heur=min_manhattan_distance,exp=5),None,['RTA_exp5'])
#show(RTA(instance_set[1],heur=min_manhattan_distance,exp=1),None,['RTA_exp1'])
#rta=RTA(heur=min_manhattan_distance,exp=20)
#show(rta.solve(instance_set[1]),None,['RTA_exp20'])
#show(RTA(instance_set[1],heur=lambda x: minimin(x,4,min_manhattan_distance),exp=20),None,['RTA_exp20'])
#show(LRTA(instance_set[1],heur=min_manhattan_distance,update_h=False),['LRTA'])
#path,hvals=LRTA(instance_set[1],heur=lambda x: minimin(x,3,min_manhattan_distance),update_h=True,iters=5)
#path,hvals=LRTA(instance_set[2],heur=lambda x: min_manhattan_distance(x),update_h=True,iters=3)
#show(path,hvals,['LRTA'])
#
#am=make_Astar(heur=min_manhattan_distance)
#for p,hs in zip(path,hvals):
#    path,stat=am(p)
#    print 'h*='+str(len(path)),hs
#


#show(LRTA(instance_set[1],heur=magsize,update_h=True),['LRTA'])
#show(LRTA(instance_set[2],heur=magsize,update_h=True),['LRTA'])
#show(LRTA(instance_set[1],heur=lambda x: 2.5*min_manhattan_distance(x)),['LRTA'])
#show(LRTA(instance_set[1],heur=lambda x: magsize(x)),['LRTA+mag'])
#show(LRTA(instance_set[2],heur=lambda x: magsize(x)),['LRTA+mag'])
#show(LRTA(instance_set[1],heur=lambda x: minimin(x,3,min_manhattan_distance)),['LRTA'])
#show(RTA(instance_set[1],heur=min_manhattan_distance),['RTA'])
#show(RTA(instance_set[0],heur=lambda x: perfecth(x,0.0,0)),['RTA'])
##test_plan_instance(4)
#test_plan_tasks_instance(6)
## instance 3 - landmark
## instance 5 - immediate loops
#for i in range(0,len(instance_set)):
#    test_plan_tasks_instance(i)

#def make_Astar(heur=zeroh,calcF=make_fCalc(),is_stop=lambda x:False, shuffle=False,lapse_rate=0,search_lapse=0):
#astar1=make_Astar(search_lapse=0.4)
#astar2=make_Astar(search_lapse=0.2)
#astar3=make_Astar(lapse_rate=0.1)
#test_log_likelihood(make_Astar(),instance_set_easy[1])
#test_log_likelihood(astar1,instance_set_easy[1])
#test_log_likelihood(astar2,instance_set_easy[1])
#test_log_likelihood(astar3,instance_set_easy[1])
#
#write_distribuition_raw(ins,make_Astar(),sample_size=100)
#write_distribuition_raw(ins,astar1,sample_size=100)
#write_distribuition_raw(ins,astar2,sample_size=100)
#write_distribuition_raw(ins,astar3,sample_size=100)

