import sys
import pickle
import os
import time
import numpy as np
import heapq as heap
import matplotlib.pyplot as plt
import scipy.stats as st
from queue import PriorityQueue
from random import sample
from random import random
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm





class Node:
    
    def __init__(self, node_id):
        self.id = node_id
        self.edges = dict()
        
    def __str__(self):
        return f"{self.id}: " + ", ".join([f"({k: v)}" for k,v in self.edges])
        
    def get_id(self):
        return self.id
    
    def get_neighbors(self):
        return [n for n in self.edges]
    
    def get_weight(self, neighbor):
        return self.edges[neighbor]
        
    def add_edge(self, neighbor, weight):
        self.edges[neighbor] = weight
        
class Graph:
    
    def __init__(self):
        self.nodes = dict()
        
    def add_node(self, node_id):
        if node_id in self.nodes:
            return
        
        node = Node(node_id)
        self.nodes[node_id] = node
        
    def get_node(self, node_id):
        return self.nodes[node_id]
    
    def add_edge(self, src, dst, cost):
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)
        
        self.nodes[src].add_edge(self.nodes[dst], cost)
        self.nodes[dst].add_edge(self.nodes[src], cost)
        
    def get_nodes(self):
        return [k for k in self.nodes]
    
    
def parse(filename='input.txt'):
    f = open(filename, 'r')
    text = f.read().strip().split('\n')
    
    G = Graph()
    
    for l in text:
        split = l.split()
        
        G.add_node(split[0])
        
        for e in split[1:]:
            n, w = e.split(':')
            
            G.add_edge(split[0], n, int(w))
    
    return G

# Implementation of Dijkstras Algorithm
def get_min_dist(graph, node):
    dists = dict()
    
    for g in graph.nodes:
        dists[g] = float('inf')
        
    dists[node] = 0
        
    queue = PriorityQueue()
    
    queue.put((0, node))
    
    while not queue.empty():
        
        cur_dist, cur_node = queue.get()
        
        if cur_dist > dists[cur_node]:
            continue
        
        for neighbor in graph.get_node(cur_node).edges:
            cost = graph.get_node(cur_node).edges[neighbor]
            
            dist = cur_dist + cost
            
            if dist < dists[neighbor.get_id()]:
                dists[neighbor.get_id()] = dist
                queue.put((dist, neighbor.get_id()))
        
        
    return dists

def get_most_central_nodes(graph, Ai, num_nodes, avg_dists = None):

    if avg_dists == None:
        avg_dists = dict()
        for n in tqdm(Ai):
            
            node_dists = get_min_dist(graph, n)
            
            avg_dists[n] = sum([node_dists[v] for v in node_dists])/len(graph.get_nodes())
        
        
    return (set(sorted(Ai, key = lambda n: avg_dists[n])[:num_nodes]), avg_dists)
        
def get_or_create_avg_dists(graph):
    if os.path.isfile('avg_dists_road.pickle'):
        return pickle.load(open('avg_dists_road.pickle', 'rb'))
    else:
        avg_dists = dict()
        for n in tqdm(graph.get_nodes()):
            
            node_dists = get_min_dist(graph, n)
            
            avg_dists[n] = sum([node_dists[v] for v in node_dists])/len(graph.get_nodes())
        
        pickle.dump(avg_dists, open('avg_dists_road.pickle', 'wb'))
        return avg_dists

# Preprocess a graph into a Distance Oracle according to the algorithm
# described by Thorup and Zwick
def preprocess(G, k = 3):
    
    # Get the nodes from the graph
    nodes = G.get_nodes()
    n = len(nodes)
  
    # First, create the subsets of decreasing size
    A = list()
    
    # A[0] contains every vertex in the graph
    A.append(set(nodes))
    
    #avg_dists = get_or_create_avg_dists(G)
    # A[1] to A[k-1] contains every element of the previous set with
    # probability n^(-1/k)
    for i in range(1, k):
        # Use a binomial distribution to determine the size of the sampling
        sample_size = np.random.binomial(len(A[i-1]), len(A[0])**(-1/(k)))
        
        # Use builtin method to pick random vertices from previous sets
        #A.append(set(sample(A[i-1], sample_size)))
        
        # Get the most connected vertices
        #prev = sorted(A[i-1], key=lambda x: len(G.get_node(x).get_neighbors()))
        #A.append(set(prev[-sample_size:]))
        
        # Get the most central vertices
        #prev = sorted(A[i-1], key=lambda x: avg_dists[x])
        #A.append(set(prev[:sample_size]))
        
        # Find j centers
        Ai = set()
        cur = sample(A[i-1], 1)[0]
        Ai.add(cur)
        dists = {key: float('inf') for key in A[i-1]}
        while len(Ai) < sample_size:
            
            for key, v in get_min_dist(G, cur).items():
                
                if key in dists: 
                    dists[key] = min(dists[key], v)
            
            
            max_dist = float('-inf')
            cur = None
            for key, v in dists.items():
                if v > max_dist:
                    max_dist = v
                    cur = key
            
            Ai.add(cur)
            
        A.append(Ai)
            
        
    # A[k] is the empty set
    A.append(set())
    #del avg_dists
    
    # If we have zero nodes in the k-1'th set, abort
    if len(A[k-1]) == 0:
        return (None, None, None)
    
    # Init vars for the oracle
    delta = defaultdict(lambda: float('inf'))
    delta_Ai = [dict() for i in range(k+1)]
    p = [dict() for i in range(k+1)]
    C = {v: set() for v in G.get_nodes()}
    B = {v: set() for v in G.get_nodes()}
    
    
    for v in G.get_nodes():
        # Every node has a distance of zero to itself
        delta[(v,v)] = 0
        
        # And a distance of infinity to the closest node i A[k]
        delta_Ai[k][v] = float('inf')
    
    # Now, for every k-1 to 0 in descending order
    for i in range(k-1, -1, -1):
        
        # Calculate delta(Ai, v) and P(v) for every v
        delta_Ai_i, p_i = get_p(G, A[i])
        delta_Ai[i] = delta_Ai_i
        p[i] = p_i
        
        # Get the cluster for every node and calculate distances
        delta_i, C_i = get_clusters(G, A, C, delta, delta_Ai, i)
        delta |= delta_i
        C |= C_i
        
    # Compute the bunch for every node
    for v in C:
        for w in C[v]:
            B[w].add(v)
            
    # Return the oracle
    return (dict(delta), B, p)
        
            
# Function for getting closets sampled node in a layer to every node
def get_p(G, A_i):
        
    # Init priority queue
    queue = []
    
    # Set to maintian seen nodes
    seen = set()
    
    # Init the vars to save info
    delta_Ai_i = {v: float('inf') for v in G.get_nodes()}
    p_i = dict()
    
    # Add all nodes sampled in the layer to the priority queue
    for n in A_i:
        heap.heappush(queue, (0, n))
        # Every sampled node has delta_Ai = 0
        delta_Ai_i[n] = 0
        p_i[n] = n
        

    # Run a dijkstra from all nodes in A_i to find closest sampled nodes to
    # all vertices
    while len(queue) > 0:
        
        cur_dist, cur_node = heap.heappop(queue)
        seen.add(cur_node)
        
        for n in G.get_node(cur_node).edges:
            if n.get_id() in seen:
                continue
            
            new_dist = cur_dist + G.get_node(cur_node).edges[n]
            
            if new_dist < delta_Ai_i[n.get_id()]:
                delta_Ai_i[n.get_id()] = new_dist
                p_i[n.get_id()] = p_i[cur_node]
                heap.heappush(queue, (new_dist, n.get_id()))
           
    return (delta_Ai_i, p_i)


# Function to get clusters for every node in a layer
def get_clusters(G, A, C, delta, delta_Ai, i):
    
    # Init queue
    queue = []
    
    # For every sampled node, not in the next layer
    for w in A[i] - A[i+1]:
        
        heap.heappush(queue, (0, w))
        seen = set()
        
        # Compute the cluster
        while len(queue) > 0:
            cur_dist, cur_node = heap.heappop(queue)
            
            seen.add(cur_node)
        
            for v in G.get_node(cur_node).get_neighbors():
                new_dist = delta[(w, cur_node)] + G.get_node(cur_node).edges[v]
                
                if v.get_id() in seen:
                    continue
                
                # Only continue if new_dist is smaller than delta(A_i+1, v)
                if new_dist > delta_Ai[i+1][v.get_id()] or new_dist >= delta[(w, v.get_id())]:
                    continue
                
                delta[(w, v.get_id())] = new_dist
                C[w].add(v.get_id())
                
                heap.heappush(queue, (new_dist, v.get_id()))
                
        
    
    return (delta, C)
 
def save_data(delta, B, p):

    file_name = 'pickle.pkl'       
    with open(file_name, "wb") as file:
        pickle.dump((delta, B, p), file)
    

def load_data():
    
    file_name = 'pickle.pkl'
    with open(file_name, 'rb') as file:
        delta, B, p = pickle.load(file)
        
    return (delta, B, p)

def query(B, delta, p, u, v):
    w = u
    i = 0
    
    while w not in B[v]:
        i += 1
        u, v = v, u
        w = p[i][u]
        
    return delta[(w,u)] + delta[(w, v)]
    
#Freedman–Diaconis
def get_number_of_bins(factors):
    
    q25, q75 = np.percentile(factors, [25, 75])
    bin_width = 2 *(q75 - q25) * len(factors) ** (-1/3)
    
    if round(bin_width) == 0:
        return 100
    
    return round((max(factors) - min(factors)) / bin_width)
    
def plot_mem_time_use(mem_use_1, mem_use_2, time_use_1, time_use_2):

    
    plt.plot(range(2,500), [sum(m) for m in mem_use_1], c=(0.77, 0, 0.05))
    plt.plot(range(2,500), [sum(m) for m in mem_use_2], c=(0.12, 0.24, 1))
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of the oracle")
    plt.show()
    
    plt.plot(range(2,500), time_use_1, color=(0.77, 0, 0.05))    
    plt.plot(range(2,500), time_use_2, c=(0.12, 0.24, 1))
    plt.xlabel("k")
    plt.ylabel("seconds")
    plt.title("Time usage of the preprocessing algorithm")
    plt.show()
    
    plt.plot(range(15,500), [sum(m) for m in mem_use_1[13:]], c=(0.77, 0, 0.05))
    plt.plot(range(15,500), [sum(m) for m in mem_use_2[13:]], c=(0.12, 0.24, 1))
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of the oracle (k>20)")
    plt.show()

def get_mem_usage(delta, B, p):
    total_mem = 0
    
    delta_mem = sys.getsizeof(delta)
    
    for k in delta:
        delta_mem += sys.getsizeof(k[0])
        delta_mem += sys.getsizeof(k[1])
        delta_mem += sys.getsizeof(delta[k])
        
    B_mem = sys.getsizeof(B)
    
    for k in B:
        B_mem += sys.getsizeof(k)
        B_mem += sys.getsizeof(B[k])
        
    p_mem = sys.getsizeof(p)
    
    for l in p:
        p_mem += sys.getsizeof(l)
        
        for k in l:
            p_mem += sys.getsizeof(k)
            p_mem += sys.getsizeof(l[k])
            
    return (delta_mem, B_mem, p_mem)

mem_use = []
time_use = []
G = parse("input_roads.txt")
k = 2
while k < 101:
    print(k)
    time_start = time.time()
    delta, B, p = preprocess(G, k)
    if delta == None:
        continue
    
    time_end = time.time()
    delta = dict(delta)
    #delta, B, p = load_data()
    
    #save_data(delta, B, p)
        
    delta_mem, B_mem, p_mem = get_mem_usage(delta, B, p)
    mem_use.append((delta_mem, B_mem, p_mem))
    time_use.append(time_end - time_start)
    
    
# =============================================================================
#     appx_factors = []
#     
#     for _ in tqdm(range(10000)):
#         u, v = sample(G.get_nodes(), 2)
#     
#         dists = get_min_dist(G, u)    
#         
#         approx = query(B, delta, p, u, v)
#         
#         appx_factors.append(approx/dists[v])
#         
#         if approx/dists[v] > (2*k)-1:
#             print(u)
#             print(v)
#             print(approx/dists[v])
#         
#     #print(stretchSum / len(list(combinations(G.get_nodes(), 2))))
#     #print(stretchSum / 10000)   
#     
#     flierprops = dict(marker='o', markerfacecolor=(0.77, 0, 0.05))
#     medianprops = dict(color=(0.77, 0, 0.05))
#     meanlineprops = dict(linestyle='-', color=(0.12, 0.24, 1))
#     
#     plt.boxplot(appx_factors, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, showmeans=True, meanline=True)
#     plt.title(f'Line Graph, Sample Centers, k={k}')
#     plt.savefig(f'Box, Line Graph, Sample Centers, k={k}.png', bbox_inches='tight')
#     plt.show()
#     plt.hist(appx_factors, bins=get_number_of_bins(appx_factors), label = 'Approximation Factors', color=(0.77, 0, 0.05))
#     mn, mx = plt.xlim()
#     plt.xlim(mn, mx)
#     plt.legend(loc = 'upper right')
#     plt.xlabel('Approximation Factors')
#     plt.title(f'Line Graph, Sample Centers, k={k}')
#     plt.savefig(f'Hist, Line Graph, Sample Centers, k={k}.png', bbox_inches='tight')
#     plt.show()
# 
# =============================================================================
    k += 1
