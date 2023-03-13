import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
from scipy.spatial import Delaunay

#Freedman–Diaconis
def get_number_of_bins(factors):
    
    q25, q75 = np.percentile(factors, [25, 75])
    bin_width = 2 *(q75 - q25) * len(factors) ** (-1/3)
    return round((max(factors) - min(factors)) / bin_width)
    

def generateId(last=''):
    
    if last == '':
        return 'A'
    
    if ord(last[-1]) < ord('Z'):
        return last[:-1] + chr(ord(last[-1])+1)
    
    return generateId(last[:-1]) + 'A'

def getDistance(src, dst):
    if len(src) == 3:
        return math.sqrt((src[0]-dst[0])**2 + (src[1]-dst[1])**2 + (src[2]-dst[2])**2)
    
    elif len(src) == 2:
        return math.sqrt((src[0]-dst[0])**2 + (src[1]-dst[1])**2)

def generateInput(num_nodes, size, seed=42, connectivity=0.5):
    
    f = open("input.txt", 'a')
    
    random.seed(seed)
    
    nodes = set()
    
    last = ''
    neighbors = dict()
    
    for n in tqdm(range(num_nodes)):
        
        x = random.randint(0, size)
        y = random.randint(0, size)
        z = random.randint(0, size)
        node_id = generateId(last)
        last = node_id
        
        node = (node_id, x, y, z)
        neighbors[node_id] = 0
        string = node[0]
        
        for dst in nodes:
            
            if random.random() <= connectivity:
                
                string += ' ' + dst[0] + ':' + str(round(getDistance(node[1:], dst[1:])))
                neighbors[node[0]] += 1
                neighbors[dst[0]] += 1
                
        nodes.add(node)
                
                
        f.write(string + '\n')
        
        
    return neighbors

def generateClusteredGraph(num_nodes, num_edges, size, seed=42):
    
    nodes = dict()
    node_coordinates = dict()
    
    random.seed(seed)

    last = ''
    
    for _ in tqdm(range(num_nodes)):
        
        x = random.randint(0, size)
        y = random.randint(0, size)
        z = random.randint(0, size)
        node_id = generateId(last)
        last = node_id
        
        node_coordinates[node_id] = (x, y, z)
        nodes[node_id] = set()
        
    unvisited = set(nodes)
    
    cur = sample(unvisited, 1)[0]
    unvisited.remove(cur)
    cur_edges = 0
    
    while len(unvisited) > 0:
        
        neighbor = sample(unvisited, 1)[0]
        unvisited.remove(neighbor)
        
        nodes[cur].add(neighbor)
        nodes[neighbor].add(cur)
        cur_edges += 1
        
        cur = neighbor
        
    choices = list(nodes)
    total_degree = sum([len(nodes[n]) for n in nodes])
    
    while cur_edges < num_edges:
        
        weights = [len(nodes[n])/total_degree for n in choices]
        
        u, v = np.random.choice(choices, 2, p=weights, replace=False)
        u, v = str(u), str(v)
        
        if u in nodes[v] or v in nodes[u]:
            continue
        
        nodes[u].add(v)
        nodes[v].add(u)
        cur_edges += 1
        total_degree += 2
        
    f = open("input.txt", "a")    
    
    for u in nodes:
        
        string = u
        
        for v in nodes[u]:
            string += ' ' + v + ':' + str(round(getDistance(node_coordinates[u], node_coordinates[v])))
            
        f.write(string + '\n')    
        
    return nodes

def generatePlanarGraph(num_nodes, size):
    
    node_coordinates = dict()
    nodes = dict()
    
    last = ''
    
    for n in range(num_nodes):
        
        x = random.randint(0, size)
        y = random.randint(0, size)
        
        node_coordinates[n] = (x, y)
        
    tri = Delaunay([node_coordinates[n] for n in node_coordinates])
    
    for t in tri.simplices:
        
        if int(t[0]) not in nodes:
            nodes[int(t[0])] = set()
        
        if int(t[1]) not in nodes:
            nodes[int(t[1])] = set()
        
        if int(t[2]) not in nodes:
            nodes[int(t[2])] = set()
        
        nodes[int(t[0])].add(int(t[1]))
        nodes[int(t[0])].add(int(t[2]))
        
        nodes[int(t[1])].add(int(t[0]))
        nodes[int(t[1])].add(int(t[2]))
        
        nodes[int(t[2])].add(int(t[0]))
        nodes[int(t[2])].add(int(t[1]))
        
        
    f = open("input.txt", "a")    
    
    for u in nodes:
        string = str(u)
        
        for v in nodes[u]:
            
            string += ' ' + str(v) + ':' + str(round(getDistance(node_coordinates[u], node_coordinates[v])))
            
        f.write(string + '\n')    
        
    return nodes

    

def generateLineGraph(num_nodes):
    
    f = open("input.txt", "a")
    
    last = ''
    
    node_id = generateId()
    
    f.write(node_id + '\n')
    
    for n in range(1, num_nodes):
        
        last = node_id
        
        node_id = generateId(node_id)
        f.write(node_id + ' '  + last + ":1\n")
   
def generateRoadGraph():
    
    edges = open("CAL_Edges.cedge").read().split('\n')
    
    f = open("input_roads.txt", "a")
    
    dists = dict()
    nodes = dict()
    for e in edges:
        
        _, n1, n2, d = e.split(' ')
        
        if n1 not in nodes:
            nodes[n1] = 0
        
        if n2 not in nodes:
            nodes[n2] = 0
            
        nodes[n1] += 1
        nodes[n2] += 1
        
        if n1 not in dists:
            dists[n1] = dict()
            
        dists[n1][n2] = d.replace('.', '')
        
    for n1 in dists:
        string = n1
        
        for n2 in dists[n1]:
            string += ' ' + n2 + ":" + dists[n1][n2]
            
        f.write(string + '\n')

    return nodes
        
        
if os.path.exists("input.txt"):
    os.remove("input.txt")

#nodes = generateInput(1000, 1000, connectivity=0.05)
#nodes = generateClusteredGraph(1000, 50000, 1000)
#nodes = generatePlanarGraph(1000, 1000)
nodes = generateLineGraph(1000)

plt.hist([nodes[n] for n in nodes], label = 'Approximation Factors', color=(0.77, 0, 0.05))
mn, mx = plt.xlim()
plt.xlim(mn, mx)
plt.legend(loc = 'upper right')
plt.xlabel('Vertex Degree')
plt.title(f'Californian Roads')
plt.show()