from collections import defaultdict
from graphviz import Digraph
from operator import itemgetter
from copy import deepcopy
#from try_pygame import *
from random import choice,shuffle
from os import path
from itertools import product
from terminal_states import terms_by_ins
from os import listdir
from os.path import join, isfile
from json import dump,load


"""
board_hash = [[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]]
"""
def drop_illegal_combos(all_combinations):
    filtered_combos = []
    for combo in all_combinations:
        hash_board = init_hash_board()
        for vehicle in combo:
            places = find_places_on_hash_board(combo, vehicle)
            if board_hash[places] == 0:
                board_hash[places] = vehicle
                filtered_comobos.append(combo)
            else:
                continue
                # we found an overlap, drop it
    return filtered_combos

def generate_all_positions_of_vehicle(vehicle, board):
    """
    Generates all positions of a vehicle on a board (if vertical slide from bottom to top...etc.)
    :param vehicle:
    :param board:
    :return:
    """
    pass

def generate_all_positions(board):
    v_positions = [generate_all_positions_of_vehicle(vehicle, board)  for vehicle in board]
    all_combinations = itertools.product(*v_positions)
    return drop_illegal_combos(all_combinations)



def rhstring(instance):
    """
    This is not a unique string, it is used only for instances of the same problem (e.g., different solutions paths)
    """
    d={}
    for p,cls in instance.h.items(): #only c changes
        d[p]=cls[0]
    for p,cls in instance.v.items(): #only l changes
        d[p]=cls[1]
    return "".join([str(d[p]) for p in sorted(d)])

#class RHInstance:
#    length,height=6,6
#    def __init__(self,h,v,name='',goal_car='r',goal_loc=[(4,2,2)]):
#        self.h=h
#        self.v=v
#        self.name=name
#        self.goal_car=goal_car
#        self.goal_loc=goal_loc
#    def __eq__(self, other):
#        if isinstance(other, RHInstance):
#            return rhstring(self)==rhstring(other)
#    def is_goal(self):
#        st_piece,oriantation = find_piece(self,self.goal_car)
#        return st_piece in self.goal_loc
#    def __ne__(self, other):
#        return (not self.__eq__(other))
#    def __hash__(self):
#        return hash(rhstring(self))
#    def __repr__(self):
#        return self.__str__()
#    def __str__(self):
#        return '({},{})'.format(self.h,self.v)
#


class RHInstance:
    length,height=6,6
    def __init__(self,h,v,name='',goal_car='r',goal_loc=[(4,2,2)]):
        self.h=h
        self.v=v
        self.name=name
        self.goal_car=goal_car
        self.goal_loc=goal_loc
    def __eq__(self, other):
        if isinstance(other, RHInstance):
            return (self.h == other.h) and (self.v== other.v)
        else:
            return False
    def is_goal(self):
        st_piece,oriantation = find_piece(self,self.goal_car)
        return st_piece in self.goal_loc
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __lt__(self, other):
        return False
    def __hash__(self):
        fs=frozenset(self.h.items() | self.v.items())
        return hash(fs)
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return '({},{})'.format(self.h,self.v)


def mag_pairs(mg):
    return set([(k,k1) for k,v in mg.items() for k1 in v ])

def unblocking_successors(instance):
    unblocking=[]
    others=[]
    mag,nodes=constuct_mag(instance)
    all_succ=expand(instance)
    for s in all_succ:
        mag_s,nodes_s=constuct_mag(s)
        if mag_pairs(mag_s)==mag_pairs(mag):
            others.append(s)
        else:
            unblocking.append(s)
    return unblocking,others

def goal_action(instance):
    """return a plan if a simple one exists"""
    for ins in expand(instance):
        if ins.is_goal():
            return ins
    return None

def plan_correction(path,plan_correction_level):
    """
    TODO: add the level
    """
    new_path=[]
    for p in path:
        new_path.append(p)
        ga=goal_action(p)
        if ga is not None and not p.is_goal():
            new_path.append(ga)
            return new_path
    return new_path


def rand_move_extended(instance,unblocking=True):
    #TODO: not going back to the previous state
    #TODO: If goal is the next step, do it. (TEST_THIS)
    ub,b=unblocking_successors(instance)
    if len([is_goal(ns) for ns in ub+b])>0:
        return ns
    if unblocking:
        return choice(ub)
    return choice(expand(instance))


def rand_move(instance,unblocking=True):
    ub,b=unblocking_successors(instance)
    if unblocking and len(ub)>0:
        return choice(ub)
    return choice(expand(instance))

def rand_play(instance,path_length=30,allow_repeats=False):
    insc=deepcopy(instance)
    path=[insc]
    while not is_goal(insc) and len(path)<path_length:
        ub,b=unblocking_successors(insc)
        insc_s=choice(ub)
        added=False
        if allow_repeats or insc_s not in path:
            path.append(insc_s)
            added=True
        if added:
            insc=insc_s
    return path

def order_tasks(instance,tasks):
    mag,nodes=constuct_mag(instance)
    order= dfs_task_order_static(instance.goal_car,mag,set([instance.goal_car]))
    #print order
    ret= [(p, tasks[p]) for p in order if p != instance.goal_car ]
    return ret

def find_tasks(instance):
    tasks={}
    terminals=find_terminal_states(instance)
    t_state = choice(terminals)
    for ins_o in (instance.h,instance.v):
        for p,cls in ins_o.items():
            if p == instance.goal_car:
                continue
            tasks[p]=[find_piece(t_state,p)[0]]
    order=order_tasks(instance,tasks)
    order.append((instance.goal_car,instance.goal_loc))
    return order



#def find_tasks(instance):
#    """ start with red on (4,2,2)
#    """
#    if instance.name=='easy4':
#        return [('g',[(1,0,2),(1,1,2),(1,2,2)]),('b',[(0,5,3)]),('o',[(0,4,2),(1,4,2)]),('p',[(5,3,3)]),('y',[(3,3,3)]),('r',[(4,2,2)])]
#    if instance.name=='easy3':
#        return [('r',[(3,2,2)]),('r',[(4,2,2)])]
#    if instance.name=='easy5':
#        return [('b',[(2,4,3)]),('r',[(4,2,2)])]
#    if instance.name=='easy6':
#        return [('p',[(5,3,3)]),('r',[(4,2,2)])]
#

def check_instance(h,v):
    all_occupied=set()
    for p,cls in h.items():
        new_ones=set(move2occupiedtiles(cls,'h'))
        if len(new_ones & all_occupied)>0:
            return False
        all_occupied |= new_ones
    for p,cls in v.items():
        new_ones=set(move2occupiedtiles(cls,'v'))
        if len(new_ones & all_occupied)>0:
            return False
        all_occupied |= new_ones
    return True


def h_unblocked(instance,level=1):
    return len(find_constraints(instance.goal_car,instance,instance.goal_loc[0]))

def min_manhattan_distance_calc(instance):
    if instance.name not in terms_by_ins:
        terms_by_ins[instance.name]=[(t.h,t.v) for t in find_terminal_states(instance)]
    return min_manhattan_distance(instance)

def min_manhattan_distance(instance,terms=None):
    # Terms is terminal states
    if terms is None:
        terms = terms_by_ins[instance.name]
    return min([manhattan_distance(instance,i) for i in terms])


def manhattan_distance(instance1, instance_pair):
    # Instance pair is terminal state
    md=0
    for cars1,cars2 in zip([instance1.h, instance1.v],instance_pair):
        for piece,loc1 in cars1.items():
            loc2 = cars2[piece]
            if loc1!=loc2:
                md+=1
    return md



#TODO: Support many locations
def find_terminal_states(instance):
    """
    Return all possible terminal states of instance.
    """
    terminals=[]
    goal_car,goal_loc=instance.goal_car,instance.goal_loc
    goal_o = find_oriantation(instance,goal_car)
    if goal_car=='r' and goal_loc[0]==(4,2,2):
        goal_loc=[(3,2,3)]
    h_options = dict((piece,piece_possible_locations(cls[0],cls[1],cls[2],'h',instance,include_current_location=True)) for piece,cls in instance.h.items() if piece != instance.goal_car )
    v_options = dict((piece,piece_possible_locations(cls[0],cls[1],cls[2],'v',instance,include_current_location=True)) for piece,cls in instance.v.items() if piece != instance.goal_car)
    h_list=[(piece,hlocations) for piece,hlocations in h_options.items()]
    for vals in product(*[x[1] for x in h_list]):
        h={}
        for p in range(len(h_list)):
            h[h_list[p][0]]=vals[p]
        v_list=[(piece,vlocations) for piece,vlocations in v_options.items()]
        for vals in product(*[x[1] for x in v_list]):
            v={}
            for p in range(len(v_list)):
                v[v_list[p][0]]=vals[p]
            ((h,v)[goal_o=='v'])[goal_car]=goal_loc[0] #currently supports one locations
            if check_instance(h,v):
                terminals.append(RHInstance(h,v))
    return terminals

def piece_possible_moves(instance,piece):
    succs=[]
    b=ground_instance(instance)
    cls,o = find_piece(instance,piece)
    possible_moves = piece_possible_locations(cls[0],cls[1],cls[2],o,instance,True)
    for m_c,m_l,m_s in possible_moves:
        if len([1 for x,y in find_path_to_location(cls,(m_c,m_l,m_s),o) if b[y][x] != ' '])==0:
            succs.append((m_c,m_l,m_s))
    return succs

def expand(instance,do_shuffle=True):
    """
    all possible moves from current instance
    """
    succs=[]
    b=ground_instance(instance)
    for car,cls in instance.h.items():
        possible_moves = piece_possible_locations(cls[0],cls[1],cls[2],'h',instance)
        for m_c,m_l,m_s in possible_moves:
            if len([1 for x,y in find_path_to_location(cls,(m_c,m_l,m_s),'h') if b[y][x] != ' '])==0:
                suc=deepcopy(instance)
                do_move(suc,(car,m_c-cls[0]))
                succs.append(suc)
    for car,cls in instance.v.items():
        possible_moves = piece_possible_locations(cls[0],cls[1],cls[2],'v',instance)
        for m_c,m_l,m_s in possible_moves:
            if len([1 for x,y in find_path_to_location(cls,(m_c,m_l,m_s),'v') if b[y][x] != ' '])==0:
                suc=deepcopy(instance)
                do_move(suc,(car,m_l-cls[1]))
                succs.append(suc)
    if do_shuffle: shuffle(succs)
    return succs


def ground_instance(instance):
    """
    returns a list of strings representing the instance
    """
    b=[[' ']*instance.length for _ in range(instance.height)]
    for piece,vals in instance.h.items():
        h_char,h_line,h_size = vals
        for hs in range(h_size):
            b[int(h_line)][h_char+hs]=piece
    for piece,vals in instance.v.items():
        v_char,v_line,v_size = vals
        for vs in range(v_size):
            b[int(v_line)+vs][v_char]=piece
    return b

def draw(instance):
    b= ground_instance(instance)
    print(' _'+"".join([str(_) for _ in range(instance.length)]) + '_')
    i=0
    for l in b:
        print(str(i)+'|'+"".join(l)+'|')
        i+=1
    print(' -'+"".join([str(_) for _ in range(instance.length)]) + '-')

def find_oriantation(instance,n):
    if n in instance.v:
        return 'v'
    elif n in instance.h:
        return 'h'
    raise(Exception)

def find_piece(instance,piece):
    st_piece =instance.v.get(piece,None)
    oriantation='v'
    if st_piece is None:
        st_piece=instance.h[piece]
        oriantation='h'
    return st_piece,oriantation

def do_move_from_fixed(instance,move):
    piece,(nc,nl,ns)=move
    cls,o=find_piece(instance,piece)
    if o == 'v':
        instance.v[piece]=(nc,nl,cls[2])
    else:
        instance.h[piece]=(nc,nl,cls[2])


def do_move(instance,move):
    piece,shift=move
    cls,o=find_piece(instance,piece)
    if o == 'v':
        instance.v[piece]=(cls[0],cls[1]+shift,cls[2])
    else:
        instance.h[piece]=(cls[0]+shift,cls[1],cls[2])

def piece_possible_locations(c,l,s,o,instance,include_current_location=False):
    L,H=instance.length,instance.height
    if o=='h':
        sq_before = sum([b_s for b_c,b_l,b_s in instance.h.values() if b_l == l and b_c<c])
        sq_after = sum([a_s for a_c,a_l,a_s in instance.h.values() if a_l == l and a_c>c])
        return [(nc,l,s) for nc in range(sq_before,L-s-sq_after+1) if nc != c or include_current_location]
    sq_before = sum([b_s for b_c,b_l,b_s in instance.v.values() if b_c == c and b_l<l])
    sq_after = sum([a_s for a_c,a_l,a_s in instance.v.values() if a_c == c and a_l>l])
    return [(c,nl,s) for nl in range(sq_before,H-s-sq_after+1) if nl != l or include_current_location]


def piece_possible_locations_find(instance,n):
    cls,o=find_piece(instance,n)
    return piece_possible_locations(cls[0],cls[1],cls[2],o,instance),o

def calc_nodes(mag,instance):
    nodes = defaultdict(dict) #{'node': {location: (coverage, (blocked_by1,...))}
    pieces=set([k for X in mag for k in mag[X] ])
    for p in pieces:
        locations = dict([(k,v) for k,v in find_coverage(p,instance,mag).items() if v>0])
        for l in locations:
            B = find_constraints(p,instance,l)  #(n,(x,y))
            locations[l]=(locations[l],[bn for bn,_ in B])
            nodes[p]=locations
    return nodes

def goal_dummy_constraints(instance):
    cls,o=find_piece(instance,instance.goal_car)
    if o=='h':
        all_locs=set([(nc,cls[1]) for nc in range(instance.length)])
    else:
        all_locs=set([(cls[0],nl) for nl in range(instance.height)])
    allowed = set([x for otl in instance.goal_loc for x in move2occupiedtiles(otl,o) ])
    goal_constraints = all_locs - allowed
    return goal_constraints

def constuct_mag(instance):
    mag = defaultdict(dict) #{'node':{'node2':[(x,y),(x2,y2)...]}}
    nodes = defaultdict(dict) #{'node': {location: (coverage, (blocked_by1,...))} {'y': {(4, 3, 3): (1.0, ['g'])}
    mag['dummy']={instance.goal_car:list(goal_dummy_constraints(instance))}
    closed = set([])
    q = [instance.goal_car]
    while len(q)>0:
        n=q.pop()
        if n in closed:
            continue
        closed.add(n)
        locations = dict([(k,v) for k,v in find_coverage(n,instance,mag).items() if v>0])
        constraints=[]
        for l in locations:
            B = find_constraints(n,instance,l)  #(n,(x,y))
            constraints.append(B)
            locations[l]=(locations[l],[bn for bn,_ in B])
            nodes[n]=locations
        if len([_ for _ in constraints if len(_)==0])>0: #has unconstraint move
            continue
        for B in constraints:
            for nn,nconst in B:
                if nn not in mag[n]:
                    closed.discard(nn)
                    mag[n][nn]=[nconst]
                elif nconst not in mag[n][nn]:
                    mag[n][nn].append(nconst)
                if nn not in q:
                    q.append(nn)
    return mag,nodes


def move2occupiedtiles(move_m,oriantation):
    if oriantation=='h':
        return [(move_m[0]+i,move_m[1]) for i in range(move_m[2])]
    else:
        return [(move_m[0],move_m[1]+i) for i in range(move_m[2])]


def find_coverage(n,instance,mag):
    """
    returns a dict of move:coverage
    """
    locs={}
    constraint_sets = [v[n] for k,v in mag.items() if n in v]
    moves,o=piece_possible_locations_find(instance,n)
    for move_m in moves:
        locs[move_m]=0
        for cs in constraint_sets:
            if len(set(move2occupiedtiles(move_m,o)) & set(cs))==0:
                locs[move_m]+=1.0/len(constraint_sets)
    return locs


def find_path_to_location(current_location,move_loc,oriantation):
    c,l,s=current_location
    c = int(c)
    l = int(l)
    s = int(s)
    move_loc_c, move_loc_l,_ = move_loc
    if oriantation=='h':
        if move_loc_c > c:
            path = [(i,l) for i in range(c+s,move_loc_c+s)]
        else:
            path = [(i,l) for i in range(move_loc_c,c)]
    else:
        if move_loc_l > l:
            path = [(c,i) for i in range(l+s,move_loc_l+s)]
        else:
            path = [(c,i) for i in range(move_loc_l,l)]
    #print 'found path for piece from '+str(current_location)+ ' to ' +str(move_loc)+ ': '+str(path)
    return path


def find_constraints(n,instance,loc):
    """
    return a set of (n,(x,y)) pairs, where x,y is the constraint that needs to be satisfied and n is the piece id that prevents it
    """
    constraints=[]
    st_piece,o=find_piece(instance,n)
    b=ground_instance(instance)
    path=find_path_to_location(st_piece,loc,o)
    for x,y in path:
        if b[int(y)][int(x)] != ' ':
            constraints.append((b[int(y)][int(x)],(x,y)))
    return constraints

def dfs_task_order_static(piece,mag,closed):
    order=[]
    for succ in mag[piece]:
        if succ not in closed:
            closed.add(succ)
            order += dfs_task_order_static(succ,mag,closed)
    closed.add(piece)
    return order+[piece]



def mag2dot(mag,view=False):
    dot = Digraph()
    ind=0
    for n in mag:
        #dot.node(n,n)
        for n1,c in mag[n].items():
            if n not in ['r', 'dummy']:
                nn = str(int(n)+1)
            else:
                nn = n
            if n1 not in ['r', 'dummy']:
                nn1 = str(int(n1)+1)
            else:
                nn1 = n1
            dot.edge(nn,nn1)#,label=str(c))
        while 1:
            dotfile='tmp{}.gv'.format(ind)
            if not path.isfile(dotfile):
                break
            ind+=1
    print(dotfile)
    dot.render(dotfile, view=view)

def is_possible_via_plan(instance,n,blocked_by,move,in_plan):
    """
    Checks if performing 'move' on piece n is possible due to moving all 'blocked_by' pieces in in_plan
    """
    if len(blocked_by)==0: return True
    cls,o = find_piece(instance,n)
    npath=find_path_to_location(cls,move,o) #[(x1,y1)..]
    for blocking_piece in blocked_by:
        bmove=in_plan.get(blocking_piece,None)
        if bmove is None: return False
        tiles_of_blocked_piece=move2occupiedtiles(bmove,find_oriantation(instance,blocking_piece)) #[(x1,y1),(x2,y2)...]
        if len(set(tiles_of_blocked_piece) & set(npath))>0:
            return False
    return True

def plan_location(instance,visits,n,nodes,in_plan):
    print('Searching location for node:{}. In_plan:{} nodes:{} visits:{}'.format(n,in_plan,nodes,visits))
    potentials = sorted([(move,(coverage,blocked_by)) for move,(coverage,blocked_by) in nodes[n].items()],key=lambda x: x[1][0], reverse=True)
    for move,(coverage,blocked_by) in potentials:
        if is_possible_via_plan(instance,n,blocked_by,move,in_plan):
            if visits[n]>= 1.0/coverage:
                print('node {} returning move: {} visits:{} ratio:{}'.format(n,move,visits[n],1.0/coverage))
                return move
    return None


#TODO: in_plan nodes are less probable
def select_search_succs(mag,visits,nodes,n):
    coverage_goal=1.0/visits[n]
    covering = list(set([tuple(succs) for move, (coverage, succs) in nodes[n].items() if coverage==coverage_goal]))
    if len(covering)==0: return mag[n]
    ret=list(choice(covering))
    shuffle(ret)
    print('possible succs for node {} are {} - choice is {}'.format(n,covering,ret))
    return ret

def search(instance,mag,nodes,n,visits,in_plan,path_so_far,keep_safe=False):
    print('calling search on {}'.format(n))
    if n in in_plan:
        print('{} already on plan'.format(n))
        return []
    if visits[n] > len([1 for X in mag.values() for k in X if k==n]) +1:
        return []
    visits[n]+=1
    next_location=plan_location(instance,visits,n,nodes,in_plan) #optimize via in_plan=None
    print('searching for possible location for node {} - {}'.format(n,next_location))
    if next_location is not None:
        in_plan[n]=next_location
        print('returning {} to {}'.format(n,next_location))
        return [(n,next_location)]
    if len(mag[n])==0 and len([1 for move, (coverage, succs) in nodes[n].items() if coverage<1])>0:
        return search(instance,mag,nodes,n,visits,in_plan,path_so_far)
    plan=[]
    for s in select_search_succs(mag,visits,nodes,n): #selection might be wrong (if it's an OR)
        plan = search(instance,mag,nodes,s,visits,in_plan,path_so_far) + plan
    if n in in_plan:
        print('{} already on plan. Returning {}'.format(n,plan))
        return plan
    next_location=plan_location(instance,visits,n,nodes,in_plan)
    print('after succs: searching for possible location for node {} - {}'.format(n,next_location))
    if next_location is not None:
        in_plan[n]=next_location
        print('adding to plan: {} to {}'.format(n,next_location))
        plan = [(n,next_location)]+plan
    print('Returning {}'.format(plan))
    return plan


def mag2plan(instance,mag,nodes,path_so_far,keep_safe=False):
    """
    Uses "insights": (a) partial move (visits/coverage) (b) red (c) loops (d) safety
    """
    return search(instance,mag,nodes,instance.goal_car,defaultdict(int),{},path_so_far)

def verify_plan(instance, tmp_plan,returned_plan):
    partial_plan=[]
    insc=deepcopy(instance)
    m = None
    for m in reversed(tmp_plan):
        print('verifing move '+str(m))
        if not verify_move_for_plan(insc,m,returned_plan):
            break
        do_move_from_fixed(insc,m)
        partial_plan.append(m)
        print('Partial plan:'+str(partial_plan))
        draw(insc)
    return m,insc,partial_plan


def verify_move_noloop(piece,instance,cls,plan_so_far):
    True

def verify_move_feasible(piece,instance,cls):
    return len(find_constraints(piece,instance,cls))==0

def verify_move_for_plan(instance,piece_move,plan_so_far):
    piece,cls=piece_move
    return verify_move_feasible(piece,instance,cls)


def verify_move(instance,move):
    piece,shift=move
    cls,o=find_piece(instance,piece)
    if o=='h':
        if (cls[0]+shift+cls[2]) >= instance.length or c+shift<0: return False
        move_loc=(cls[0]+shift,cls[1],cls[2])
    else:
        if ((cls[1]+shift+cls[2]) >= instance.height or cls[1]+shift<0): return False
        move_loc=(cls[0],cls[1]+shift,cls[2])
    return len(find_constraints(piece,instance,move_loc))==0


def mag_in_edges(mag):
    return set([(to_node,l) for x in [v for k,v in current_mag.items() if k != 'dummy'] for (to_node,locs) in x.items() for l in locs ])


def learn_from_mag(current_mag,last_mag,instance):
    print('current_mag:{}'.format(current_mag))
    print('last_mag:{}'.format(last_mag))
    new_mag=deepcopy(current_mag)
    current_pairs= set([(from_node,to_node,l) for from_node in current_mag if from_node != 'dummy' for (to_node,locs) in current_mag[from_node].items() for l in locs ])
    to_nodes=set([tn[1] for tn in current_pairs ])
    last_pairs = set([(from_node,to_node,l) for from_node in last_mag if from_node != 'dummy' for (to_node,locs) in last_mag[from_node].items() for l in locs ])
    print('current_pairs:{}'.format(current_pairs))
    print('last_pairs:{}'.format(last_pairs))
    possible_learns = last_pairs - current_pairs
    for f,n,l in possible_learns:
        if n in to_nodes:
            print('learning \'dummy\':({},{})'.format(n,l))
            nconsts = new_mag['dummy.'+f].get(n,[])
            nconsts.append(l)
            new_mag['dummy.'+f][n]=nconsts
    return new_mag,calc_nodes(new_mag,instance)

def plan_with_tasks(instance):
    print('Starting to plan+tasks for instance:{}'.format(instance.name))
    tasks = find_tasks(instance)
    print('Subtasks for instance:{}'.format(tasks))
    returned_plan=[]
    solving_instance=deepcopy(instance)
    for t,t_locs in tasks:
        solving_instance.goal_car=t
        solving_instance.goal_loc=t_locs
        print('Subtasks- moving {} to {}'.format(t,t_locs))
        while 1:
            if solving_instance.is_goal():
                break
            mag,nodes=constuct_mag(solving_instance)
            print('starting to plan for node {} from:'.format(t))
            draw(solving_instance)
            print('MAG:{}'.format(mag))
            print('NODES:{}'.format(nodes))
            mag2dot(mag,view=False)
            tmp_plan=mag2plan(solving_instance,mag,nodes,returned_plan)
            print(' pending plan:{}'.format(tmp_plan))
            violating_move,solving_instance,partial_plan=verify_plan(solving_instance,tmp_plan,returned_plan)
            returned_plan.extend(partial_plan)
    return returned_plan


def plan(instance):
    print('Starting to plan for instance:{}'.format(instance.name))
    solving_instance=deepcopy(instance)
    returned_plan=[]
    last_mag={}
    while 1:
        current_mag,nodes=constuct_mag(solving_instance)
        mag,nodes = learn_from_mag(current_mag,last_mag,solving_instance)
        last_mag= current_mag
        print('starting to plan from:')
        draw(solving_instance)
        print('MAG:{}'.format(mag))
        print('NODES:{}'.format(nodes))
        mag2dot(mag,view=False)
        tmp_plan=mag2plan(solving_instance,mag,nodes,returned_plan)
        print(' pending plan:{}'.format(tmp_plan))
        violating_move,solving_instance,partial_plan=verify_plan(solving_instance,tmp_plan,returned_plan)
        returned_plan.extend(partial_plan)
        if solving_instance.is_goal():
            print('**solved instance!**')
            return returned_plan


def make_path_for_plan_model(ins,path_moves):
    path=[ins]
    insc=deepcopy(ins)
    for m in path_moves:
        insc=deepcopy(insc)
        do_move_from_fixed(insc,m)
        path.append(insc)
    path.reverse()
    return path


def make_instance(lines):
    colors='bwlBycMgmoptnsG'
    h,v={},{}
    name=lines[0].rstrip()
    size=lines[1].rstrip()
    red_car = lines[2].rstrip().split()
    h['r']=(int(red_car[0]),int(red_car[1]),int(red_car[3]))
    i=0
    for line in lines[3:]:
        car = line.rstrip().split()
        ((h,v)[car[2]=='v'])[colors[i]]=(int(car[0]),int(car[1]),int(car[3]))
        i+=1
    return RHInstance(h,v,name)

def read_instances(filename='../princeton_AI/code/jams.txt'):
    """
    read instances according to the format in https://www.cs.princeton.edu/courses/archive/fall11/cos402/assignments/programs/rushhour/
    """
    instance_set=[]
    with open(filename,'r') as insf:
        contents=insf.readlines()
    ins_begin=0
    for i in range(len(contents)):
        if contents[i].rstrip()=='.':
            instance_set.append(make_instance(contents[ins_begin:i]))
            ins_begin=i+1
    return instance_set

def json_to_ins(filename):
    with open(filename,'r') as data_file:
        data = load(data_file)
        h={}
        v={}

        ins=RHInstance(h,v,name=data['id'])
        for c  in data['cars']:
            cars=(h,v)[c['orientation']=='vertical']
            cars[c['id']]=((int(c['position'])%6),(int(c['position'])/6),c['length'])
    return ins


def read_instances_json(json_dir='../psiturk-rushhour/static/json'):
    instance_set=[]
    jsons=[join(json_dir, f) for f in listdir(json_dir) if f.endswith('.json') and isfile(join(json_dir, f))]
    for f in jsons:
        instance_set.append(json_to_ins(f))
    return instance_set


def instance_dict():
    return dict((x.name, x) for x in read_instances_json())

opt_solution_instances_mag={ 0:9, 1:10, 2:15, 3:10, 4:10, 5:11, 6:14, 7:14, 8:13, 9:21, 10:27, 11:19, 12:24, 13:18, 14:24, 15:28, 16:28, 17:27, 18:23, 19:13, 20:22, 21:29, 22:30, 23:29, 24:41, 25:32, 26:34, 27:33, 28:34, 29:38, 30:39, 31:39, 32:49, 33:45, 34:48, 35:46, 36:49, 37:50, 38:58, 39:58}


opt_solution_instances= {'Jam-29': 31, 'Jam-28': 30, 'Jam-21': 21, 'Jam-20': 10, 'Jam-23': 29, 'Jam-22': 26, 'Jam-25': 27, 'Jam-24': 25, 'Jam-27': 28, 'Jam-26': 28, 'Jam-2': 8, 'Jam-3': 14, 'Jam-1': 8, 'Jam-6': 9, 'Jam-7': 13, 'Jam-4': 9, 'Jam-5': 9, 'Jam-8': 12, 'Jam-9': 12, 'Jam-40': 51, 'Jam-14': 17, 'Jam-15': 23, 'Jam-16': 21, 'Jam-17': 24, 'Jam-10': 17, 'Jam-11': 25, 'Jam-12': 17, 'Jam-13': 16, 'Jam-18': 25, 'Jam-19': 22, 'Jam-38': 48, 'Jam-39': 50, 'Jam-32': 37, 'Jam-33': 40, 'Jam-30': 32, 'Jam-31': 37, 'Jam-36': 44, 'Jam-37': 47, 'Jam-34': 43, 'Jam-35': 43}
