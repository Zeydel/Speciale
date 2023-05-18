import sys
import pickle
import os
import time
import math
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
import networkx as nx

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

class TreeNode:
    
    def __init__(self, sequence):
        self.sequence = sequence
        self.j = None
        self.max_delta = float('-inf')
    
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
        A.append(set(sample(A[i-1], sample_size)))
        
        # Get the most connected vertices
        #prev = sorted(A[i-1], key=lambda x: len(G.get_node(x).get_neighbors()))
        #A.append(set(prev[-sample_size:]))
        
        # Get the most central vertices
        #prev = sorted(A[i-1], key=lambda x: avg_dists[x])
        #A.append(set(prev[:sample_size]))
        
        # Find j centers
# =============================================================================
#         Ai = set()
#         cur = sample(A[i-1], 1)[0]
#         Ai.add(cur)
#         dists = {key: float('inf') for key in A[i-1]}
#         while len(Ai) < sample_size:
#             
#             for key, v in get_min_dist(G, cur).items():
#                 
#                 if key in dists: 
#                     dists[key] = min(dists[key], v)
#             
#             
#             max_dist = float('-inf')
#             cur = None
#             for key, v in dists.items():
#                 if v > max_dist:
#                     max_dist = v
#                     cur = key
#             
#             Ai.add(cur)
#             
#         A.append(Ai)
# =============================================================================
            
        
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
 
# =============================================================================
# def get_d(G, k, delta, p):
#     
#     d = list()
#     
#     for i in range(0, k-2):
#         d.append(dict())
#         for u in G.get_nodes():
#             v1 = delta[(p[i+2][u], u)]
#             v2 = delta[(p[i][u], u)]
#             d[i][u] = v1 - v2
#             
#     return d
# =============================================================================
    
def build_T(I):
    
    T = [-1]
    
    T.append(TreeNode(I))
    
    i = 1
    
    while i < len(T):
        if T[i] == None:
            i += 1
            continue
        if len(T[i].sequence) > 2: 
            T.append(TreeNode(T[i].sequence[:len(T[i].sequence)//2+1]))
            T.append(TreeNode(T[i].sequence[len(T[i].sequence)//2:]))
        else:
            T.append(None)
            T.append(None)
            
        i += 1
    
    return T

def enrich_T(T, delta, p, u):
    
    for i in range(len(T)-1, 0, -1):
        
        if T[i] == None:
            continue
        
        if len(T[i].sequence) == 2:
            
            for j in T[i].sequence:
                
                if delta[(p[j+2][u], u)] - delta[(p[j][u], u)] > T[i].max_delta:
                    T[i].max_delta = delta[(p[j+2][u], u)] - delta[(p[j][u], u)]
                    T[i].j = j
            continue
    
        if T[i*2].max_delta > T[(i*2)+1].max_delta:
            T[i].max_delta = T[i*2].max_delta
            T[i].j = T[i*2].j
        else:
            T[i].max_delta = T[(i*2)+1].max_delta
            T[i].j = T[(i*2)+1].j
            
    return T
    
def get_s(T, i1, i2, i = 1):
    S = set()
    
    if T[i].sequence[0] == i1 and T[i].sequence[-1] == i2:
        S.add(T[i])
    
    elif i1 in T[i*2].sequence and i2 in T[i*2].sequence:
        S |= get_s(T, i1, i2, i*2)
    
    elif i1 in T[(i*2)+1].sequence and i2 in T[(i*2)+1].sequence:
        S |= get_s(T, i1, i2, (i*2)+1)
        
    elif i1 in T[i*2].sequence and i2 in T[(i*2)+1].sequence:
        S |= get_s(T, i1, T[i*2].sequence[-1], i*2)
        S |= get_s(T, T[(i*2)+1].sequence[0], i2, (i*2)+1)
    
    return S

def get_depth(idx):
    
    return math.floor(math.log2(idx))

def get_s_new(T, i1, i2):
    
    S = set()
    
    a, b = 1, 1
    
    while len(T[a].sequence) > 2:
        if i1 >= T[(a*2)+1].sequence[0] and i1 <= T[(a*2)+1].sequence[-1]:
            a = (a*2) +1
        else:
            a = a*2
            
    while len(T[b].sequence) > 2:
        if i2 >= T[b*2].sequence[0] and i2 <= T[b*2].sequence[-1]:
            b = b*2
        else:
            b = (b*2)+1
    
    while get_depth(a) > get_depth(b):
        if a % 2 == 1:
            S.add(T[a])
            a += 1
        a = a//2
        
    while get_depth(b) > get_depth(a):
        if b % 2 == 0:
            S.add(T[b])
            b -= 1
        b = b//2
    
    while a <= b:
        if a % 2 == 1:
            S.add(T[a])
            a += 1
        if b % 2 == 0:
            S.add(T[b])
            b -= 1
            
        a = a//2
        b = b//2
        
    return S
        
        
def get_j(T, i1, i2):

    S = get_s_new(T, i1, i2)
        
    j = None
    max_delta = float('-inf')
    
    for s in S:
        
        if s.max_delta > max_delta:
            max_delta = s.max_delta
            j = s.j
            
    return j
        
    
    
    
def get_d(G, k, delta, p):
    
    d = dict()
    
    for u in G.get_nodes():
        I = [i for i in range(k-2) if i % 2 == 0]
        T = build_T(I)
        T = enrich_T(T, delta, p, u)
        
        d[u] = dict()    
        
        nodes = set()
        nodes.add(1)
        
        while len(nodes) > 0:
            cur = nodes.pop()
            
            if T[cur] == None:
                continue
            
            d[u][(T[cur].sequence[0], T[cur].sequence[-1])] = T[cur].j
            
            if cur*2 < len(T):
                nodes.add(cur*2)
                
            if (cur*2)+1 < len(T):
                nodes.add((cur*2)+1)
                
        
        for i1 in range(0, k-2, 2):
            low = i1 + math.floor(math.log2(k)//2)
            
            if low % 2 == 1:
                low += 1
            
            d[u][(low, low)] = low
            
            for i2 in range(low, k-2, 2):
                
                if i2 - i1 > (k//2) + 1:
                    break
                
                if (i1, i2) not in d[u]:
                    d[u][(i1, i2)] = get_j(T, i1, i2)
                
    return d


def get_d_new(G, k, delta, p):
    
    d_max = dict()
    d = dict()
    
    
    for u in G.get_nodes():
    
        d[u] = dict()
        d_max[u] = dict()
    
        for i in range(0, k-4, 2):
            if delta[(p[i+2][u], u)] - delta[(p[i][u], u)] > delta[(p[i+4][u], u)] - delta[(p[i+2][u], u)]:
                d_max[u][(i, i+2)] = delta[(p[i+2][u], u)] - delta[(p[i][u], u)]
                d[u][(i, i+2)] = i
            else:
                d_max[u][(i, i+2)] = delta[(p[i+4][u], u)] - delta[(p[i+2][u], u)]
                d[u][(i, i+2)] = i+2
        
        for l in range(2, (k//2) + 1):
            for i1 in range(0, k-2-(2*l), 2):
                
                i2 = i1 + (2*l)
                
                
                i = (i1 + i2) // 2
                if i % 2 == 1:
                    i += 1
                
                if d_max[u][i1, i] > d_max[u][i, i2]:
                    d_max[u][i1, i2] = d_max[u][i1, i]
                    d[u][i1,i2] = d[u][i1, i]
                else:
                    d_max[u][i1, i2] = d_max[u][i, i2]
                    d[u][i1, i2] = d[u][i, i2]
            
                    
    return d
            
def get_d_naive(G, k, delta, p):

    d = dict()
    
    for u in G.get_nodes():
        
        d[u] = dict()    
        
        nodes = set()
        nodes.add(1)
                        
        
        for i1 in range(0, k-2, 2):
            low = i1 + math.floor(math.log2(k)//2)
            
            if low % 2 == 1:
                low += 1
            
            d[u][(low, low)] = low
            
            for i2 in range(low, k-4, 2):
                
                if i2 - i1 > (k//2) + 1:
                    break
                
                if (i1, i2) not in d[u]:
                    
                    for i in range(i1, i2+1, 2):
                        if delta[(p[i+2][u], u)] - delta[(p[i][u], u)] > delta[(p[i+4][u], u)] - delta[(p[i+2][u], u)]:
                            d[(i1, i2)] = i
                    
                
    return d
      
            
        
        
    
 
def save_data(delta, B, p):

    file_name = 'pickle.pkl'       
    with open(file_name, "wb") as file:
        pickle.dump((delta, B, p), file)
    

def load_data():
    
    file_name = 'pickle.pkl'
    with open(file_name, 'rb') as file:
        delta, B, p = pickle.load(file)
        
    return (delta, B, p)

def bquery(B, delta, p, k, u, v, i1, i2):
    if i2 - i1 <= math.log2(k):
        return query(B, delta, p, u, v, i1)

    i = (i1 + i2) // 2
    
    if i % 2 == 1:
        i += 1
    
    j = d[u][(i1, i-2)]
    
            
    if p[j][u] not in B[v] and p[j+1][v] not in B[u]:
        return bquery(B, delta, p, k, u, v, i, i2)
    else:
        return bquery(B, delta, p, k, u, v, i1, j)
    
def bquery_new(B, delta, p, k, u, v, i1, i2):
    if i2 - i1 <= math.log2(k):
        return query(B, delta, p, u, v, i1)

    i = (i1 + i2) // 2
    
    if i % 2 == 1:
        i += 1
                
    if p[i][u] not in B[v] and p[i+1][v] not in B[u]:
        return bquery_new(B, delta, p, k, u, v, i, i2)
    else:
        return bquery_new(B, delta, p, k, u, v, i1, i)

def query(B, delta, p, u, v, i = 0):
    w = u
    
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
    
def plot_mem_time_use(mem_uses, time_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    #max_len = max([len(m) for m in time_uses])
# =============================================================================
#     
#     for i in range(len(time_uses)):
#         #mem_uses[i] = mem_uses[i] + ([None] * (max_len-len(mem_uses[i])))
#         time_uses[i] = time_uses[i] + ([None] * (max_len-len(time_uses[i])))
#     
# # =============================================================================
# #     for i, mem_use in enumerate(mem_uses):
# #         plt.plot(range(16,500), [None if m == None else m for m in mem_use], c=colors[i])
# #     plt.ylim(0, 2000000000)
# #     plt.xlabel("k")
# #     plt.ylabel("bytes")
# #     plt.title("Memory usage of the oracle")
# #     plt.show()
# #     
# # =============================================================================
#     for i, time_use in enumerate(time_uses):
#         plt.plot(range(16,max_len+16), time_use, c=colors[i])    
#     plt.xlabel("k")
#     plt.ylabel("Seconds")
#     plt.title("Time usage of the preprocessing algorithm")
#     plt.show()
# =============================================================================
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(16,101), [None if m == None else sum(m) for m in mem_use], c=colors[i])
    plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of the oracle")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(16,101), [None if m == None else m[0] for m in mem_use], c=colors[i])
    plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of delta")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(16,101), [None if m == None else m[1] for m in mem_use], c=colors[i])
    plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of B")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(16,101), [None if m == None else m[2] for m in mem_use], c=colors[i])
    plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of p")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(16,101), [None if m == None else m[3] for m in mem_use], c=colors[i])
    plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of d")
    plt.show()
    

def get_mem_usage(delta, B, p, d):
    B_mem = sys.getsizeof(B)
        
    for k in B:
        B_mem += sys.getsizeof(k)
        B_mem += sys.getsizeof(B[k])
            
        for v in B[k]:
            B_mem += sys.getsizeof(v)
                
    p_mem = sys.getsizeof(p)
        
    for k in p:
        p_mem += sys.getsizeof(k)
            
        for v in k:
            p_mem += sys.getsizeof(v)
            p_mem += sys.getsizeof(k[v])

    delta_mem = sys.getsizeof(delta)
        
    for k in delta:
        delta_mem += sys.getsizeof(k[0])
        delta_mem += sys.getsizeof(k[1])
        delta_mem += sys.getsizeof(delta[k])

        
    d_mem = sys.getsizeof(d)    
    
    for u in d:
        d_mem += sys.getsizeof(u)
        
        for p in d[u]:
            d_mem += sys.getsizeof(p[0])
        
    return (delta_mem, B_mem, p_mem, d_mem)

def is_planar(G):
    
    nxG = nx.Graph()
    
    for v in G.get_nodes():
        nxG.add_node(v)
        
    for v in G.get_nodes():
        for e in G.get_node(v).get_neighbors():
            nxG.add_edge(v, e.id)
            
    return nx.is_planar(nxG)

mem_use = []
time_use = []
time_use_with_d = []

G = parse("input_roads.txt")
k = 16
print(is_planar(G))

times = []
times_fast = []

appx_factors = []
appx_factors_new = []

while k < 17:
    print(k)
    time_start = time.time()
    delta, B, p = preprocess(G, k)
    if delta == None:
        continue
    
    time_end = time.time()
    time_use.append(time_end - time_start)
    
   
    
    d = get_d(G, k, delta, p)
    time_use_with_d.append(time.time() - time_start)
    
    #delta, B, p = load_data()
    
    #save_data(delta, B, p)
    delta = dict(delta)
         
    mem_use.append(get_mem_usage(delta, B, p, d))
    
    
    appx_factors_k = []
    appx_factors_new_k = []
    

    
    sample_pairs = []
    sample_pair_dists = dict()
    
    
    
    for _ in tqdm(range(50000)):
        u, v = sample(G.get_nodes(), 2)
    
        sample_pairs.append((u,v))
    
        dists = get_min_dist(G, u)    
        sample_pair_dists[(u,v)] = dists[v]
    
    start = time.time()    
    
    for u, v in sample_pairs:
        approx = bquery_new(B, delta, p, k, u, v, 0, k-1)
        appx_factors_k.append(approx/sample_pair_dists[(u,v)])
        
    end = time.time()
    time_std = end - start
            
    start = time.time()
    
    for u, v in sample_pairs:
        approx = bquery_new(B, delta, p, k, v, u, 0, k-1)
        appx_factors_new_k.append(approx/sample_pair_dists[(u,v)])
        
    end = time.time()
    
    time_fast = end-start
    times.append(time_std)
    times_fast.append(time_fast)    
    
    appx_factors.append(appx_factors_k)
    appx_factors_new.append(appx_factors_new_k)
    
    #print(stretchSum / len(list(combinations(G.get_nodes(), 2))))
    #print(stretchSum / 10000)   
    
    flierprops = dict(marker='o', markerfacecolor=(0.77, 0, 0.05))
    medianprops = dict(color=(0.77, 0, 0.05))
    meanlineprops = dict(linestyle='-', color=(0.12, 0.24, 1))
    
    plt.boxplot(appx_factors_k, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, showmeans=True, meanline=True)
    plt.title(f'Fast query with additional preprocessing with outliers, k={k}')
    plt.savefig(f'Box, Fast query with additional preprocessing with outliers, k={k}.png', bbox_inches='tight')
    plt.show()
    plt.hist(appx_factors_k, bins=get_number_of_bins(appx_factors_k), label = 'Approximation Factors', color=(0.77, 0, 0.05))
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    plt.legend(loc = 'upper right')
    plt.xlabel('Approximation Factors')
    plt.title(f'Fast query with additional preprocessing, k={k}')
    plt.savefig(f'Hist, Fast query with additional preprocessing, k={k}.png', bbox_inches='tight')
    plt.show()
    
    plt.boxplot(appx_factors_new_k, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, showmeans=True, meanline=True)
    plt.title(f'Fast query without additional preprocessing with outliers, k={k}')
    plt.savefig(f'Box, Fast query without additional preprocessing with outliers, k={k}.png', bbox_inches='tight')
    plt.show()
    plt.hist(appx_factors_new_k, bins=get_number_of_bins(appx_factors_new_k), label = 'Approximation Factors', color=(0.77, 0, 0.05))
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    plt.legend(loc = 'upper right')
    plt.xlabel('Approximation Factors')
    plt.title(f'Fast query without additional preprocessing, k={k}')
    plt.savefig(f'Hist, Fast query without additional preprocessing, k={k}.png', bbox_inches='tight')
    plt.show()
    k += 1
