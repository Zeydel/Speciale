from __future__ import print_function
import matplotlib.pyplot as plt
import math
import sys
import time
import numpy as np
import heapq as heap
import pickle
from collections import defaultdict
from random import sample, random
from queue import PriorityQueue
from tqdm import tqdm
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

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

class Oracle:
    
    def __init__(self, k):
        self.k = k
        self.B = None
        self.p = None
        self.delta = None
        self.D = None
        
        self.queryTime = 0.0
        self.preprocessingTime = 0.0
        
    def init_simple_oracle(self, G, k):
        
        start = time.time()
        
        nodes = G.get_nodes()
        n = len(nodes)
        
        A = list()
        
        A.append(set(nodes))
        
        for i in range(1,k):
            
            sample_size = np.random.binomial(len(A[i-1]), len(A[0])**(-1/k))
            A.append(set(sample(A[i-1], sample_size)))
            
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
    
    def init_d(self):
        
        start = time.time()
        
        self.D = dict()

        for u in G.get_nodes():
            I = [i for i in range(k-2) if i % 2 == 0]
            T = self.build_T(I)
            T = self.enrich_T(T, u)
            
            self.D[u] = dict()    

            subsequences = set()
            subsequences.add((0, self.k-1))
            
            while len(subsequences) > 0:
                
                cur = subsequences.pop()
                
                if cur[1] - cur[0] <= math.log2(k):
                    continue
                
                i = (cur[0] + cur[1]) // 2
            
                if i % 2 == 1:
                    i += 1

                j = self.get_j(T, cur[0], i-2)
                
                self.D[u][(cur[0], i-2)] = j
                subsequences.add((cur[0], j))
                subsequences.add((i, cur[1]))
            
        self.preprocessingTime += time.time() - start
          
        
    def init_d_dump(self):
        
        start = time.time()
        
        self.D = dict()

        for u in G.get_nodes():
            
            self.D[u] = dict()    

            subsequences = set()
            subsequences.add((0, self.k-1))
            
            while len(subsequences) > 0:
                
                cur = subsequences.pop()
                
                if cur[1] - cur[0] <= math.log2(k):
                    continue
                
                i = (cur[0] + cur[1]) // 2
            
                if i % 2 == 1:
                    i += 1

                j = -float('inf')
                for l in range(cur[0], i-3, 2):
                    if self.delta[(self.p[l+2][u], u)] - self.delta[(self.p[l][u], u)] > j:
                        j = l
                                    
                self.D[u][(cur[0], i-2)] = j
                subsequences.add((cur[0], j))
                subsequences.add((i, cur[1]))
            
        self.preprocessingTime += time.time() - start
    
    def init_d_naive(self):
        start = time.time()
        
        self.D = dict()

    
        d_max = dict()
        
        
        for u in G.get_nodes():
        
            self.D[u] = dict()
            d_max[u] = dict()
        
            for i in range(0, k-4, 2):
                if self.delta[(self.p[i+2][u], u)] - self.delta[(self.p[i][u], u)] > self.delta[(self.p[i+4][u], u)] - self.delta[(self.p[i+2][u], u)]:
                    d_max[u][(i, i+2)] = self.delta[(self.p[i+2][u], u)] - self.delta[(self.p[i][u], u)]
                    self.D[u][(i, i+2)] = i
                else:
                    d_max[u][(i, i+2)] = self.delta[(self.p[i+4][u], u)] - self.delta[(self.p[i+2][u], u)]
                    self.D[u][(i, i+2)] = i+2
            
            for l in range(2, (k//2) + 1):
                for i1 in range(0, k-2-(2*l), 2):
                    
                    i2 = i1 + (2*l)
                    
                    
                    i = (i1 + i2) // 2
                    if i % 2 == 1:
                        i += 1
                    
                    if d_max[u][i1, i] > d_max[u][i, i2]:
                        d_max[u][i1, i2] = d_max[u][i1, i]
                        self.D[u][i1,i2] = self.D[u][i1, i]
                    else:
                        d_max[u][i1, i2] = d_max[u][i, i2]
                        self.D[u][i1, i2] = self.D[u][i, i2]
                
                        
        self.preprocessingTime += time.time() - start

    
    def get_j(self, T, i1, i2):

        S = self.get_s(T, i1, i2)
        
        if i1 == i2:
            return i2
        
        j = None
        max_delta = float('-inf')
        
        for s in S:
            
            if s.max_delta > max_delta:
                max_delta = s.max_delta
                j = s.j
                
        return j
    
    def build_T(self, I):
        
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
    
    def get_s(self, T, i1, i2):
    
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
        
        while self.get_depth(a) > self.get_depth(b):
            if a % 2 == 1:
                S.add(T[a])
                a += 1
            a = a//2
            
        while self.get_depth(b) > self.get_depth(a):
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
    
    def get_depth(self, idx):    
        return math.floor(math.log2(idx))
    
    def enrich_T(self, T, u):
    
        for i in range(len(T)-1, 0, -1):
            
            if T[i] == None:
                continue
            
            if len(T[i].sequence) == 2:
                
                for j in T[i].sequence:
                    
                    if self.delta[(self.p[j+2][u], u)] - self.delta[(self.p[j][u], u)] > T[i].max_delta:
                        T[i].max_delta = self.delta[(self.p[j+2][u], u)] - self.delta[(self.p[j][u], u)]
                        T[i].j = j
                continue
        
            if T[i*2].max_delta > T[(i*2)+1].max_delta:
                T[i].max_delta = T[i*2].max_delta
                T[i].j = T[i*2].j
            else:
                T[i].max_delta = T[(i*2)+1].max_delta
                T[i].j = T[(i*2)+1].j
                
        return T
       
     
    def query(self, u, v, i1, i2):
        if i2 - i1 <= math.log2(k):
            return self.dist_k(u, v, i1)
    
        i = (i1 + i2) // 2
        
        if i % 2 == 1:
            i += 1
        
        j = self.D[u][(i1, i-2)]
        
                
        if self.p[j][u] not in self.B[v] and self.p[j+1][v] not in self.B[u]:
            return self.query(u, v, i, i2)
        else:
            return self.query(u, v, i1, j)
     
    def dist_k(self, u, v, i = 0):
        w = self.p[i][u]
                        
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
            
        D_mem = sys.getsizeof(self.D)
        
        for k in self.D:
            D_mem += sys.getsizeof(k)
            D_mem += sys.getsizeof(self.D[k])
            
            for v in self.D[k]:
                D_mem += sys.getsizeof(v[0])
                D_mem += sys.getsizeof(v[1])
                D_mem += sys.getsizeof(self.D[k][v])
        
        return (B_mem, p_mem, delta_mem, D_mem)
        
    
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
    for i, time_use in enumerate(time_uses):
        plt.plot(range(5,76), time_use, c=colors[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage of the preprocessing algorithm")
    plt.show()
# =============================================================================
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(5,76), [None if m == None else sum(m) for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of the oracle")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(5,76), [None if m == None else m[0] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of B")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(5,76), [None if m == None else m[1] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of p")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(5,76), [None if m == None else m[2] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of delta")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(5,76), [None if m == None else m[3] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of D")
#     plt.show()
# =============================================================================
    


G = parse("input_roads.txt")
sample_pair_dists = dict()
appx_factors = []

mem_uses = []
query_time_uses = []
preprocessing_time_uses = []

for k in range(5, 201):
    print(k)
    O = Oracle(k)
    print('Oracle Initialised')
    O.init_simple_oracle(G, k)
    print('B, p, and delta Initialised')
    O.init_d_dump()
    print('D Initialised')    

    sample_pairs = []

    preprocessing_time_uses.append(O.preprocessingTime)    
    mem_uses.append(O.get_memory_usage())
# =============================================================================
# 
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
#         approx = O.query(u, v, 0, k-1)
#         approx_factors_k.append(approx/sample_pair_dists[(u,v)])
#     
#     O.queryTime = time.time() - start
#     appx_factors.append(approx_factors_k)    
#     query_time_uses.append(O.queryTime)
#     print(O.queryTime)
# =============================================================================
    del O
    
with open('Final_data/D_construct/naive_m_long', 'wb') as file:
    pickle.dump(mem_uses, file)
    
with open('Final_data/D_construct/naive_p_long', 'wb') as file:
    pickle.dump(preprocessing_time_uses, file)
    
