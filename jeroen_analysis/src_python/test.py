import numpy as np
import graphviz

g = graphviz.Digraph('test')
g.attr(compound='true')
g.attr(labelloc='b')

with g.subgraph(name='cluster_0') as c:
    c.attr(label='Layer 0')
    c.node('R', style='filled', fillcolor='#ff0000')

with g.subgraph(name='cluster_1') as c:
    c.attr(label='Layer 1')
    c.node('1')
    c.node('2')
    c.node('7')

with g.subgraph(name='cluster_2') as c:
    c.attr(label='Layer 2')
    c.node('3')
    c.node('8')

with g.subgraph(name='cluster_3') as c:
    c.attr(label='Layer 3')
    c.node('5')
    c.node('6')

with g.subgraph(name='cluster_4') as c:
    c.attr(label='Layer 4')
    c.node('4')

g.edge('R', '1')
g.edge('R', '2')
g.edge('R', '7')

g.edge('1', '3')
g.edge('2', '8')
g.edge('7', '8')

g.edge('8', '1')
g.edge('8', '5')
g.edge('3', '6')

g.edge('5', '3')
g.edge('6', 'R')
g.edge('6', '4')

g.render()