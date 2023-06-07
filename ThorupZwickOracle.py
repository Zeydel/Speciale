from __future__ import print_function
import matplotlib.pyplot as plt
import math
import sys
import time
import pickle
import numpy as np
import heapq as heap
from collections import defaultdict
from random import sample, random
from queue import PriorityQueue
from tqdm import tqdm
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import os

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
    

class Oracle:
    
    def __init__(self, k):
        self.k = k
        self.B = None
        self.p = None
        self.delta = None
        
        self.queryTime = 0.0
        self.preprocessingTime = 0.0
        
    def init_simple_oracle(self, G, k):
        
        start = time.time()
        
        nodes = G.get_nodes()
        n = len(nodes)
        
        A = list()
        
        A.append(set(nodes))
        
        avg_dists = get_or_create_avg_dists(G)
        
        
        test = time.time()
        for i in range(1,k):
            
            sample_size = np.random.binomial(len(A[i-1]), len(A[0])**(-1/k))
            
            # Random sampling
            #A.append(set(sample(A[i-1], sample_size)))
            
            prev = sorted(A[i-1], key=lambda x: len(G.get_node(x).get_neighbors()))
            A.append(set(prev[-sample_size:]))
            
            #prev = sorted(A[i-1], key=lambda x: avg_dists[x])
            #A.append(set(prev[:sample_size]))
            
# =============================================================================
#             Ai = set()
#             cur = sample(A[i-1], 1)[0]
#             Ai.add(cur)
#             dists = {key: float('inf') for key in A[i-1]}
#             while len(Ai) < sample_size:
#                          
#                 for key, v in get_min_dist(G, cur).items():
#                              
#                     if key in dists: 
#                         dists[key] = min(dists[key], v)
#                          
#                          
#                 max_dist = float('-inf')
#                 cur = None
#                 for key, v in dists.items():
#                     if v > max_dist:
#                         max_dist = v
#                         cur = key
#                          
#                 Ai.add(cur)
#                          
#             A.append(Ai)
# =============================================================================
        print(time.time() - test)
            
            
        A.append(set())
        
        if len(A[k-1]) == 0:
            return self.init_simple_oracle(G, k)
        
        self.B = {v: set() for v in nodes}
        self.p = [dict() for i in range(k+1)]
        self.delta = defaultdict(lambda: float('inf'))
        
        delta_Ai = [dict() for i in range(k+1)]
        C = {v: set() for v in nodes}
        
        for v in nodes:
            
            self.delta[(v,v)] = 0
            
            delta_Ai[k][v] = float('inf')
        
        for i in range(k-1, -1, -1):
            
            delta_Ai_i, p_i = self.get_p(G, A[i])
            delta_Ai[i] = delta_Ai_i
            self.p[i] = p_i
            
            delta_i, C_i = self.get_clusters(G, A, C, self.delta, delta_Ai, i)
            self.delta |= delta_i
            C |= C_i
            
        for v in C:
            for w in C[v]:
                self.B[w].add(v)
                
        self.delta = dict(self.delta)
        self.preprocessingTime += time.time() - start
            
            
    def get_p(self, G, A_i):
        
        queue = []
        
        seen = set()
        
        delta_Ai_i = {v: float('inf') for v in G.get_nodes()}
        p_i = dict()
        
        for n in A_i:
            heap.heappush(queue, (0,n))
            
            delta_Ai_i[n] = 0
            p_i[n] = n
            
        
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
    
    
    def get_clusters(self, G, A, C, delta, delta_Ai, i):
        
        queue = []
        
        for w in A[i] - A[i+1]:
            
            heap.heappush(queue, (0, w))
            seen = set()
            
            while len(queue) > 0:
                cur_dist, cur_node = heap.heappop(queue)
                
                seen.add(cur_node)
                
                for v in G.get_node(cur_node).get_neighbors():
                    new_dist = delta[(w, cur_node)] + G.get_node(cur_node).edges[v]
                    
                    if v.get_id() in seen:
                        continue
                    
                    if new_dist > delta_Ai[i+1][v.get_id()] or new_dist >= delta[(w, v.get_id())]:
                        continue
                    
                    delta[(w, v.get_id())] = new_dist
                    C[w].add(v.get_id())
                    
                    heap.heappush(queue, (new_dist, v.get_id()))
        
        return (delta, C)
    
    def query(self, u, v):
        
        w = u
        i = 0
        
        while w not in self.B[v]:
            
            i += 1
            u, v = v, u
            w = self.p[i][u]
                        
        return self.delta[(w,u)] + self.delta[(w, v)] 
    
    def get_memory_usage(self):
        
        B_mem = sys.getsizeof(self.B)
        
        for k in self.B:
            B_mem += sys.getsizeof(k)
            B_mem += sys.getsizeof(self.B[k])
            
            for v in self.B[k]:
                B_mem += sys.getsizeof(v)
                
        p_mem = sys.getsizeof(self.p)
        
        for k in self.p:
            p_mem += sys.getsizeof(k)
            
            for v in k:
                p_mem += sys.getsizeof(v)
                p_mem += sys.getsizeof(k[v])

        delta_mem = sys.getsizeof(self.delta)
        
        for k in self.delta:
            delta_mem += sys.getsizeof(k[0])
            delta_mem += sys.getsizeof(k[1])
            delta_mem += sys.getsizeof(self.delta[k])
        
        return (B_mem, p_mem, delta_mem)
        
    
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

def plot_mem_time_use(mem_uses, time_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    #max_len = max([len(m) for m in time_uses])
    
    #for i in range(len(time_uses)):
        #mem_uses[i] = mem_uses[i] + ([None] * (max_len-len(mem_uses[i])))
    #    time_uses[i] = time_uses[i] + ([None] * (max_len-len(time_uses[i])))
    
# =============================================================================
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(16,500), [None if m == None else m for m in mem_use], c=colors[i])
#     plt.ylim(0, 2000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of the oracle")
#     plt.show()
#     
# =============================================================================
# =============================================================================
#     for i, time_use in enumerate(time_uses):
#         plt.plot(range(3,76), time_use, c=colors[i])    
#     plt.xlabel("k")
#     plt.ylabel("Seconds")
#     plt.title("Time usage of 50000 queries")
#     plt.show()
# =============================================================================
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(3,76), [None if m == None else sum(m) for m in mem_use], c=colors[i])
    #plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of the oracle")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(3,76), [None if m == None else m[0] for m in mem_use], c=colors[i])
    #plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of B")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(3,76), [None if m == None else m[1] for m in mem_use], c=colors[i])
    #plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of p")
    plt.show()
    
    for i, mem_use in enumerate(mem_uses):
        plt.plot(range(3,76), [None if m == None else m[2] for m in mem_use], c=colors[i])
    #plt.ylim(0, 1000000000)
    plt.xlabel("k")
    plt.ylabel("bytes")
    plt.title("Memory usage of delta")
    plt.show()


G = parse("input_roads.txt")
sample_pair_dists = dict()
appx_factors = []

mem_uses = []
query_time_uses = []
preprocessing_time_uses = []

for k in range(2, 76):
    print(k)
    O = Oracle(k)
    print('Oracle Initialised')
    O.init_simple_oracle(G, k)
    print('B, p, and delta Initialised')
    
    sample_pairs = []

    preprocessing_time_uses.append(O.preprocessingTime)    
    mem_uses.append(O.get_memory_usage())

# =============================================================================
#     samples = []
#     approx_factors_k = []
#     
#     nodes = G.get_nodes()
# 
#     for _ in range(1000):
#         
#         u, v = sample(nodes, 2)
#         
#         samples.append((u,v))
#         sample_pair_dists[(u,v)] = get_min_dist(G, u)[v]
# 
#     start = time.time()
# 
#     for u, v in samples:
# 
#         approx = O.query(u, v)
#         approx_factors_k.append(approx/sample_pair_dists[(u,v)])
#     
#     O.queryTime = time.time() - start
#     appx_factors.append(approx_factors_k)    
#     query_time_uses.append(O.queryTime)
#     print(O.queryTime)
# =============================================================================
    del O
    
with open('Final_data/Sampling_strat/centers_m', 'wb') as file:
    pickle.dump(mem_uses, file)
    
with open('Final_data/Sampling_strat/centers_p', 'wb') as file:
    pickle.dump(preprocessing_time_uses, file)
    

    