"""
Generating rushhour instances.

1) Generate a terminal state based on the car sizes
2) reject trival cases
3) Starting from the terminal state, perform quasi-random moves backwards.
4) then solve optimally forward
5) output to json

"""

#test code
from  rushhour import read_instances,draw,mag2dot, json_to_ins,opt_solution_instances,min_manhattan_distance_calc
#test code

from time import time
from itertools import product
from json import dump,load
from rushhour import RHInstance,ground_instance,do_move,rand_move, rhstring,h_unblocked,constuct_mag, mag_pairs
from astar import make_Astar
from copy import deepcopy
from sys import argv
from os import listdir
from os.path import join, isfile
from random import randint,choice,sample
from instances_selected_set4_list import puzzle_files

def number_of_possible_plans_from_mag():
    """ """
    return 999

def depth_of_mag():
    """ """
    return 999

#plans 
def number_of_optimal_plans(ins):
    """ How many optimal plans are there for this instance"""
    return 999

def added_edges_between_mags(mags_1,mag_2):
    return len(mag_pairs(mag_2) - mag_pairs(mags_1))

def get_optimal_solution(optimal_solution):
    if optimal_solution is None:
        astar = make_Astar(heur=min_manhattan_distance_calc)
        return astar(ins)
    else:
        return optimal_solution

def number_of_unsafe_moves(ins,optimal_solution=None):
    """
    Number of unsafe moves on the optimal path.
    """
    plan,nodes=get_optimal_solution(optimal_solution)
    ret=0;
    mags=[constuct_mag(s)[1] for s in plan]
    for i in range(len(mags)-1):
        if added_edges_between_mags(mags[i],mags[i+1]) > 0:
            ret+=1
    return ret

def change_in_heuristic_value(ins,optimal_solution=None):
    """Manhattan distance over path"""
    plan,nodes=get_optimal_solution(optimal_solution)
    hs=[min_manhattan_distance_calc(s) for s in plan]
    print(hs)
    return 999


def number_of_unsafe_edges(ins):
    """
    Statistics for how much edges are added to the mag
    """
    return 999

def node_decrease_rate(ins):
    """
    at what rate nodes are decreasing from the mag
    """
    return 999

def edge_decrease_rate(ins):
    """the rate of edge decrease"""
    return 999
def nodes_expanded_by_astar(ins):
    """ This is expected to correlate with change_in_heuristic_value. If the effect on humans is not as strong, people make abstractions"""
    return 999


def generate_instance(min_path_length,max_path_length,cars_2,cars_3):
    ins = None
    while ins is None:
        ins=generate_terminal_instance(cars_2,cars_3)
    play_back(ins,max_path_length)
    candidate=deepcopy(ins)
    plan=solve_forward(ins)
    if len(plan)>min_path:
        return candidate,plan

def solve_forward(ins):
    astar = make_Astar(heur=min_manhattan_distance)
    plan,stat=astar(ins)
    return plan

def possible_locations(ins,orientation,size):
    """
    returns a list of possible locations  (l,c,s) based on orientation and size
    """
    b=ground_instance(ins) #[line][column]
    ret=[]
    for c,l in product(range(ins.length), range(ins.height)):
        if orientation == 'h':
            sqrs = [b[l][cc] for cc in range(c,min(c+size,ins.length)) if b[l][cc]==' ' ]
        else:
            sqrs = [b[cl][c] for cl in range(l,min(l+size,ins.height)) if b[cl][c]==' ' ]
        if len(sqrs)==size:
            ret.append((c,l,size))
    return ret


def add_cars(ins,cars_2,cars_3):
    cars_to_place=[2]*cars_2+[3]*cars_3
    car_id=0
    while len(cars_to_place)>0:
        size=cars_to_place.pop(randint(0,len(cars_to_place)-1))
        orientation = choice(['h','v'])
        pl=possible_locations(ins,orientation,size)
        if len(pl)==0:
            return None
        cd=(ins.v,ins.h)[orientation=='h']
        cd[str(car_id)]=choice(pl)
        car_id+=1
    return ins



def generate_initial_instance(cars_2,cars_3):
    loc=choice([(i,2,2) for i in range(3)])
    ins = RHInstance({'r':loc},{},'i_id')
    return add_cars(ins,cars_2,cars_3)


def generate_terminal_instance(cars_2,cars_3):
    ins = RHInstance({'r':(4,2,2)},{},'i_id')
    return add_cars(ins,cars_2,cars_3)


def play_back_goal(ins,num_moves):
    ins.goal_loc=[choice([(i,2,2) for i in range(3)])]
    astar=make_Astar(heur=h_unblocked)
    print(ins.goal_loc)
    path,stats=astar(ins)
    print(path)
    if len(path)==0:
        return None
    return path[-1]


def play_back_rand(ins,num_moves):
    for i in range(num_moves):
        ins=rand_move(ins,False)
    return ins



def play_back_closed(ins,num_moves):
    """TODO : dont repeat symmetric moves"""
    closed_list= set([])
    for i in range(num_moves):
        while True:
            ins=rand_move(ins,False)
            if rhstring(ins) not in closed_list:
                closed_list.add(rhstring(ins))
                break
        print(len(closed_list))
        print(i)
    return ins


"""
{ "id": "problem1",
  "cars": [
    {
      "id": "c1",
      "orientation": "horizontal",
      "position": 6,
      "length": 2,
      "player": false
    },...
    ]
}
"""


def instance_to_json(ins,length):
    filename = '_'.join([str(time()).replace('.','_'),ins.name , str(length),'.json'])
    print('outputing to '+filename)
    d={'id': ins.name}
    cars = [{'id': cid, 'orientation':'vertical', 'position': 6*l+c , 'length': s ,'player': False} for cid,(c,l,s) in ins.v.iteritems()]
    cars.extend([{'id': cid, 'orientation':'horizontal', 'position': 6*l+c , 'length': s ,'player': cid=='r' } for cid,(c,l,s) in ins.h.iteritems()])
    d['cars']=cars
    with open(filename,'w') as fp:
        dump(d,fp)



def create_instances(num_of_instances,min_path_length,max_path_length,cars_2,cars_3):
    for i in range(num_of_instances):
        init_ins=generate_initial_instance(cars_2,cars_3)
        if init_ins is None:
            continue
        init_ins.name = 'prb'+str(i)
        am=make_Astar(search_limit=max_path_length,heur=h_unblocked)
        path,stat=am(init_ins)
        length=len(path)
        print(length)
        if length > min_path_length:
            instance_to_json(init_ins,length)

def instance_stat_raw(ins):
    sz=[s for c,l,s in ins.h.values()+ins.v.values()]
    car_2 = str(len([1 for s in sz if s==2]))
    car_3 = str(len([1 for s in sz if s==3]))
    v_size = str(len(ins.v))
    h_size = str(len(ins.h))
    mag,nodes=constuct_mag(ins)
    sccs=tarjanSCC(mag)
    mag_nodes=str(len(nodes))
    mag_edges=str(sum([len(nd) for nd in mag.values()]))
    num_sccs=str(len([1 for scc in sccs if len(scc)>=2]))
    max_scc_size=str(max([len(scc) for scc in sccs]))
    path_length = str(opt_solution_instances[ins.name])
    return ','.join([ins.name,car_2,car_3,v_size,h_size,mag_nodes,mag_edges,path_length,num_sccs,max_scc_size])


def instance_stat_plan(jsonfile):
    instance=jsonfile.split('_')[2]
    ins = json_to_ins(jsonfile)
    number_of_optimal_plans_ = str(number_of_optimal_plans(ins))
    number_of_unsafe_moves_ = str(number_of_unsafe_moves(ins))
    number_of_unsafe_edges_ = str(number_of_unsafe_edges(ins))
    node_decrease_rate_ = str(node_decrease_rate(ins))
    edge_decrease_rate_ = str(edge_decrease_rate(ins))
    change_in_heuristic_value_ = str(change_in_heuristic_value(ins))
    return ','.join([instance,jsonfile,number_of_optimal_plans_,number_of_unsafe_moves_,number_of_unsafe_edges_,node_decrease_rate_,edge_decrease_rate_,change_in_heuristic_value_])



def instance_stat(jsonfile):
    ins = json_to_ins(jsonfile)
    instance=jsonfile.split('_')[2]
    sz=[s for c,l,s in list(ins.h.values())+list(ins.v.values())]
    car_2 = str(len([1 for s in sz if s==2]))
    car_3 = str(len([1 for s in sz if s==3]))
    v_size = str(len(ins.v))
    h_size = str(len(ins.h))
    mag,nodes=constuct_mag(ins)
    sccs=tarjanSCC(mag)
    mag_nodes=str(len(nodes))
    mag_edges=str(sum([len(nd) for nd in mag.values()]))
    num_sccs=str(len([1 for scc in sccs if len(scc)>=2]))
    max_scc_size=str(max([len(scc) for scc in sccs]))
    path_length = jsonfile.split('_')[3]
    avg_bf=str(float(mag_edges)/len(mag.values()))
    avg_location_size=str(float(sum([len(n) for n in nodes.values()]))/len(nodes.values()))
    return ','.join([instance,jsonfile,car_2,car_3,v_size,h_size,mag_nodes,mag_edges,path_length,num_sccs,max_scc_size,avg_bf,avg_location_size])

def tarjanSCC(mag):
    sccs=[]
    index=0
    strong_connect('dummy',mag,[],[0],sccs,{})
    return sccs


def strong_connect(n,mag,s,index,sccs,indd):
    s.append(n)
    n_index=n_lowlink=index[0]
    indd[n]=(n_index,n_lowlink)
    #print '{0}{1}:({2},{3})'.format(' ',n,n_index,n_lowlink)
    for k in mag[n]:
        if k not in s:
            index[0]+=1
            k_index,k_lowlink=strong_connect(k,mag,s,index,sccs,indd)
            n_lowlink=min(n_lowlink,k_lowlink)
            indd[n]=(n_index,n_lowlink)
        else:
            n_lowlink=min(n_lowlink, indd[k][0])
            indd[n]=(n_index,n_lowlink)
    if n_index == n_lowlink:
        scc=[]
        while True:
            scc.append(s.pop())
            if scc[-1]==n:
                sccs.append(scc)
                break
    #print '{0}{1}:({2},{3})'.format(' ',n,n_index,n_lowlink)
    return n_index,n_lowlink





#test_code
#instance_set = read_instances()
#mag,nodes=constuct_mag(instance_set[15])
#mag2dot(mag,True)
#tarjanSCC(mag)
#exit()

#term_ins=generate_terminal_instance(3,3)
#draw(term_ins)
#for i in range(100):
#    ins=play_back_goal(term_ins,1000)
#    if ins is None:
#        print 'Unsolved instance:'
#        draw(term_ins)
#        exit()
#    ast=make_Astar(heur=h_unblocked)
#    path,stats=ast(ins)
#    draw(ins)
#    print 'path:'+str(len(path))
#print possible_locations(instance_set[0],'v',2)


#test_code
def pick_sample(jsons_data,size,ideal_distribution,factor_key):
    path_length_dist=ideal_distribution[factor_key]
    s = set([])
    for path_length,count in path_length_dist.items():
        s |= set(sample([x for x in range(len(jsons_data)) if jsons_data[x][factor_key]==path_length],count))
    while len(s)<size:
        c=choice(range(len(jsons_data)))
        if c not in s:
            s.append(c)
    return s

def extend_sample(jsons_data,size,ideal_distribution,factor_key,s):
    """
    """
    path_length_dist=ideal_distribution[factor_key]
    while len(s)<size:
        for path_length,count in path_length_dist.items():
            extra=count-len([1 for l in s if jsons_data[l][factor_key]==path_length])
            if extra>0:
                s |= set(sample([x for x in range(len(jsons_data)) if jsons_data[x][factor_key]==path_length],extra))
    return s

def diff_counts(counters,s,jsons_data):
    ret=0
    for factor in counters:
        for factor_value in counters[factor]:
            ret+= abs(counters[factor][factor_value]-len([1 for l in s if jsons_data[l][factor]==factor_value]))
    return ret

def drop_extras(counters,s,jsons_data,k=.25):
    """
    look for peaks and drop it
    """
    X={}
    for factor in counters:
        for factor_value in counters[factor]:
            extra = len([1 for l in s if jsons_data[l][factor]==factor_value])-counters[factor][factor_value]
            #print '({3}={4}): {0}-{1}={2}'.format(len([1 for l in s if jsons_data[l][factor]==factor_value]),counters[factor][factor_value],extra,factor,factor_value)
            if extra>0:
                s = s-set(sample([x for x in s if jsons_data[x][factor]==factor_value],extra))
        mismatched=[l for l in s if jsons_data[l][factor] not in counters[factor].keys()]
        #print 'len of mismatched:'+str(len(mismatched))
        s = s-set(sample(mismatched,int(len(mismatched)*k)))
    return s

def find_puzzle_set_ext(ideal_distribution,jsons_data,iterations,size):
    """
    ideal_distribution is a dict of counters.
    jsons_data is a list of strings
    """
    best_guess_scr = float('inf')
    best_guess=None
    s = pick_sample(jsons_data,size,ideal_distribution,'path_length')
    print('Initial size: {0}'.format(len(s)))
    for i in range(iterations):
        s = drop_extras(ideal_distribution,s,jsons_data)
        print('After dropping : {0}'.format(len(s)))
        s = extend_sample(jsons_data,size,ideal_distribution,'path_length',s)
        print('After extension : {0}'.format(len(s)))
        scr = diff_counts(ideal_distribution,s,jsons_data)
        print(scr)
        if scr < best_guess_scr:
            best_guess_scr = scr
            best_guess = s
    print('best:'+str(best_guess_scr))
    return best_guess



def find_puzzle_set(ideal_distribution,jsons_data,iterations,size):
    """
    ideal_distribution is a dict of counters.
    jsons_data is a list of strings
    """
    best_guess_scr = float('inf')
    best_guess=None
    for i in range(iterations):
        s = pick_sample(jsons_data,size,ideal_distribution,'path_length')
        scr = diff_counts(ideal_distribution,s,jsons_data)
        if scr < best_guess_scr:
            best_guess_scr = scr
            best_guess = s
    print('best:'+str(best_guess_scr))
    return best_guess

def instance_set_from_jsons(json_dir, how_many=10):
    jsons=[join(json_dir, f) for f in listdir(json_dir) if f.endswith('.json') and isfile(join(json_dir, f))]
    return [json_to_ins(f) for f in jsons[:how_many]]

def read_json_data(json_dir):
    jsons=[join(json_dir, f) for f in listdir(json_dir) if f.endswith('.json') and isfile(join(json_dir, f))]
    fields=['instance','jsonfile','car_2','car_3','v_size','h_size','mag_nodes','mag_edges','path_length','num_sccs','max_scc_size']
    data=[]
    for f in jsons:
        try:
            vals=instance_stat(f).split(',')
            data.append(dict(zip(fields,vals)))
        except Exception as e:
            print('Problem reading '+ f)
            raise e
    return data



def create_stats(json_dir,is_plan):
    fields=(['jsonfile','car_2','car_3','v_size','h_size','mag_nodes','mag_edges','path_length','num_sccs','max_scc_size','avg_bf','avg_locatio_size'],['instance','jsonfile','number_of_optimal_plans','number_of_unsafe_moves','number_of_unsafe_edges','node_decrease_rate','edge_decrease_rate','change_in_heuristic_value'])[is_plan]
    print(','.join(fields))
    if json_dir == 'raw':
        inss=read_instances()
        for f in inss:
            try:
                print(instance_stat_raw(f))
            except Exception as e:
                print('error in instance ' + f.name + str(e))
                raise e
    else:
        if json_dir == 'set4':
            inss=['../psiturk-rushhour/'+f for f in puzzle_files]
        else:
            inss=[join(json_dir, f) for f in listdir(json_dir) if f.endswith('.json') and isfile(join(json_dir, f))]
        for f in inss:
            # try:
            if is_plan:
                print(instance_stat_plan(f))
            else:
                print(instance_stat(f))
            # except Exception as e:
            #     print('error in file ' + f + str(e))

def print_stats(jsons_data,puzzle_indx):
    fields=['jsonfile','car_2','car_3','v_size','h_size','mag_nodes','mag_edges','path_length','num_sccs','max_scc_size']
    print(','.join(fields))
    for i in puzzle_indx:
        j=jsons_data[i]
        print(','.join([j[f] for f in fields]))

def main():
    try:
        command=argv[1]
        if command.startswith('stat'):
            json_dir = argv[2]
        elif command == 'find':
            json_dir = argv[2]
            iters = int(argv[3])
            for_stat = False
            if len(argv)>=4:
                for_stat = argv[4]=='for_stat'
        elif command=='generate':
            num_of_instances=int(argv[2])
            min_path_length=int(argv[3])
            max_path_length=int(argv[4])
            cars_2=int(argv[5])
            cars_3=int(argv[6])
        else:
            raise(Exception('unknown command'))
    except:
        print('usage: [stat | stat_plan | generate|find] [<json_dir> | <json_dir> | <num_of_instances> <min_path_length> <max_path_length> <cars_2> <car_3> | <json_dir> <iterations> [for_stat]] use \'set4\' if there is no json_dir')
        exit()

    if command=='stat':
        create_stats(json_dir,False)
    if command=='stat_plan':
        create_stats(json_dir,True)
    elif command=='generate':
        create_instances(num_of_instances,min_path_length,max_path_length,cars_2,cars_3)
    elif command=='find':
        jsons_data=read_json_data(json_dir)
        puzzle_indx=find_puzzle_set_ext({
            'path_length':{'7':18,'11':18,'14':17,'16':17},
            'v_size':{'2':23,'4':23,'7':24},
            'max_scc_size':{'1':24,'4':23,'8':23},
            'mag_edges':{'4':35,'16':35},
            'mag_nodes':{'4':23,'6':23,'9':24}},jsons_data,iters,70)
        print_stats(jsons_data,puzzle_indx)
        if not for_stat:
            print([jsons_data[i]['jsonfile'] for i in puzzle_indx ])

import os
if __name__ == "__main__":
    main()
