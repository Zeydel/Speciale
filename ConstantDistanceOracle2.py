from __future__ import print_function
import matplotlib.pyplot as plt
import math
import sys
import time
import numpy as np
import heapq as heap
from collections import defaultdict
from random import sample, random
from queue import PriorityQueue
from tqdm import tqdm
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass



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
        self.D = None
        self.I = None
        self.evenUp = None
        self.evenDown = None
        self.simpleOracle = None
        self.x1 = None
        self.x2 = None
        self.x3 = None
        
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
    
    
    def init_D_and_I(self, G):
        
        start = time.time()
        
        self.D = dict()
        self.I = dict()
        
        for v in G.get_nodes():
            
            D_v = dict()
            I_v = dict()
            
            I_v[0] = 0
            
            D_v[2] = self.delta[self.p[2][v], v] - self.delta[self.p[0][v], v]
            I_v[2] = 2
            
            for j in range(4, k, 2):
                
                d_j = self.delta[self.p[j][v], v] - self.delta[self.p[j-2][v], v]
                
                if d_j > D_v[j-2]:
                    D_v[j] = d_j
                    I_v[j] = j
                else:
                    D_v[j] = D_v[j-2]
                    I_v[j] = I_v[j-2]
                    
            self.D[v] = D_v
            self.I[v] = I_v
            
        self.preprocessingTime += time.time() - start
            
    def init_evens(self, G, k):
        
        start = time.time()
        
        self.evenDown = dict()
        self.evenUp = dict()
        
        for v in G.get_nodes():
            evenDown_v = dict()
            evenUp_v = dict()
            
            for w in self.B[v]:
                d_pow = rnd_pow(self.delta[w,v])
                
                if d_pow in evenUp_v:
                    continue
                
                if k % 2 == 1:
                    max_k = k-1
                else:
                    max_k = k-2
                
                for j in range(max_k, -1, -2):
                    if self.delta[self.p[j][v], v] <= d_pow:
                      evenUp_v[d_pow] = j
                      break
                 
                for j in range(0, max_k+1, 2):
                    if rnd_pow(self.delta[self.p[j][v], v]) == rnd_pow(self.delta[self.p[evenUp_v[d_pow]][v], v]):
                        evenDown_v[d_pow] = j
                        break
                
            self.evenDown[v] = evenDown_v
            self.evenUp[v] = evenUp_v
            
        self.preprocessingTime += time.time() - start
    
    def init_x(self, G, k):
        
        start = time.time()
        
        self.x1 = dict()
        self.x2 = dict()
        self.x3 = dict()
        
        for v in G.get_nodes():
            
            x1_v = dict()
            x2_v = dict()
            x3_v = dict()
            
            x1_v[0] = 0
            x2_v[0] = 0
            x3_v[0] = 0
            
            for i in range(2, k, 2):
                
                for j in range(2, k, 2):
                    
                    if (j - i) * (self.D[v][j] - self.D[v][i]) > (k - 2 - j)*self.D[v][i]:
                        x1_v[i] = j
                        break
                    
                if i not in x1_v:
                    if (k-1) % 2 == 0:
                        x1_v[i] = k-1
                    else:
                        x1_v[i] = k-2
                
                for j in range(x1_v[i], k, 2):
                    
                    if (j - x1_v[i]) * (self.D[v][j] - self.D[v][x1_v[i]]) > (k - 2 - j) * self.D[v][x1_v[i]]:
                        x2_v[i] = j
                        break
                    
                if i not in x2_v:
                    if (k-1) % 2 == 0:
                        x2_v[i] = k-1
                    else:
                        x2_v[i] = k-2
                        
                for j in range(x2_v[i], k, 2):
                    
                    if (j - x2_v[i]) * (self.D[v][j] - self.D[v][x2_v[i]]) > x1_v[i] * (self.D[v][x2_v[i]] - self.D[v][x1_v[i]]):
                        x3_v[i] = j
                        break
                
                if i not in x3_v:
                    if (k-1) % 2 == 0:
                        x3_v[i] = k-1
                    else:
                        x3_v[i] = k-2
                        
            self.x1[v] = x1_v
            self.x2[v] = x2_v
            self.x3[v] = x3_v
            
        self.preprocessingTime += time.time() - start
            
    def init_MN_oracle(self, G, k):
        
        start = time.time()
        
        k_MN = (128*k) // 2
        
        self.simpleOracle = Oracle(k_MN)
        self.simpleOracle.init_simple_oracle(G, k_MN)
        
        self.preprocessingTime += time.time() - start
        
    def simple_query(self, u, v, i = 0):
        w = self.p[i][u]
                
        
        while w not in self.B[v]:
            
            i += 1
            u, v = v, u
            w = self.p[i][u]
                        
        return self.delta[(w,u)] + self.delta[(w, v)] 
    
    def dist_k(self, u, v, i = 0):
        w = self.p[i][u]
                
        i1 = i
        
        while w not in self.B[v]:
            
            i += 1
            u, v = v, u
            w = self.p[i][u]
            
        if i - i1 > 1:
            raise Exception("To long time spent in loop")
            
        return self.delta[(w,u)] + self.delta[(w, v)] 
    
    def query(self, u, v):
        
        start = time.time()
        
        deltaMN = self.simpleOracle.simple_query(u, v)
        
        deltaMN_time = time.time() - start
        
        deltaMN_pow = rnd_pow(deltaMN)//512
        i1, i2 = None, None
        
        i1_u, i2_u = None, None
        
        i1_v, i2_v = None, None
        
        d_max, d_max_u, d_max_v = None, None, None
        
        while deltaMN_pow <= 4*rnd_pow(deltaMN):
            
            if deltaMN_pow in self.evenDown[u]:
                d_max_u = deltaMN_pow
                
                if (not self.is_terminal(self.I[u][self.evenDown[u][deltaMN_pow]]-2, u, v)) and self.is_terminal(self.I[u][self.evenUp[u][deltaMN_pow]]-2, u, v):
                    i1_u = self.evenDown[u][deltaMN_pow]
                    i2_u = self.evenUp[u][deltaMN_pow]
                    
            if deltaMN_pow in self.evenDown[v]:
                d_max_v = deltaMN_pow
            
                if (not self.is_terminal(self.I[v][self.evenDown[v][deltaMN_pow]]-2, u, v)) and self.is_terminal(self.I[v][self.evenUp[v][deltaMN_pow]]-2, u, v):
                    i1_v = self.evenDown[v][deltaMN_pow]
                    i2_v = self.evenUp[v][deltaMN_pow]
                
            deltaMN_pow *= 2
                    
        if d_max_u == None and d_max_v == None:
            return (rnd_pow(deltaMN)//256, deltaMN_time)

        if d_max_u == None:
            u, v = v, u
            i1, i2 = i1_v, i2_v
            d_max = d_max_v
        else:
            i1, i2 = i1_u, i2_u
            d_max = d_max_u
            
        
        i_max = self.evenUp[u][d_max] + 2
        
        if i_max >= k:
            
            if (k-1) % 2 == 0:
                i_max = k-1
            else:
                i_max = k-2
                
        
        if not i1 == None:
            return (self.query_legit(u, v, i1, i2), deltaMN_time)
        elif not self.is_terminal(self.I[u][i_max]-2, u, v):
            if i_max >= k-2:
                return (self.dist_k(u, v, i_max), deltaMN_time)
            else:
                return (deltaMN, deltaMN_time)
        else:
            deltaMN_pow = rnd_pow(deltaMN)//512
            
            i_min = float('inf')
            
            while deltaMN_pow <= 4*rnd_pow(deltaMN):
               
                if deltaMN_pow in self.evenDown[u] and self.I[u][self.evenDown[u][deltaMN_pow]] - 2 < i_min and self.is_terminal(self.I[u][self.evenDown[u][deltaMN_pow]] - 2, u, v):
                    i_min = self.I[u][self.evenDown[u][deltaMN_pow]] - 2
               
                deltaMN_pow *= 2
                
            if self.I[u][i_max]-2 < i_min and self.is_terminal(self.I[u][i_max]-2, u, v):
                i_min = self.I[u][i_max]-2
                
            return (self.dist_k(u, v, i_min), deltaMN_time)
            
            
    def query_legit(self, u, v, i1, i2):
        y1 = self.I[u][self.x1[u][i1]]
        y2 = self.I[u][self.x2[u][i1]]
        y3 = self.I[u][self.x3[u][i1]]
        y4 = self.I[u][i2]
                
        if self.is_terminal(y1 - 2, u, v):
            return self.dist_k(u, v, y1-2)
        if self.is_terminal(y2 - 2, u, v):
            return self.dist_k(u, v, y2-2)
        if self.is_terminal(y3 - 2, u, v):
            return self.dist_k(u, v, y3-2)
        if self.x3[u][i1] > self.k - 2:
            return self.dist_k(u, v, self.x3[u][i1])
        return self.dist_k(u, v, y4-2)
        
        
    def is_terminal(self, i, u, v):
        if i < 0:
            return False
        if i == self.k-1:
            return True
        if self.p[i][u] in self.B[v] or self.p[i+1][v] in self.B[u]:
            return True
        return False
    
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

        I_mem = sys.getsizeof(self.I)

        if self.I is not None:
            for k in self.I:
                I_mem += sys.getsizeof(k)
                I_mem += sys.getsizeof(self.I[k])
                    
                for v in self.I[k]:
                    I_mem += sys.getsizeof(v)
                    I_mem += sys.getsizeof(self.I[k][v])
        
        D_mem = sys.getsizeof(self.D)
        
        if self.D is not None:
            for k in self.D:
                D_mem += sys.getsizeof(k)
                D_mem += sys.getsizeof(self.D[k])
                
                for v in self.D[k]:
                    D_mem += sys.getsizeof(v)
                    D_mem += sys.getsizeof(self.D[k][v])

        x_mem = sys.getsizeof(self.x1)
        x_mem += sys.getsizeof(self.x2)
        x_mem += sys.getsizeof(self.x3)
        
        if self.x1 is not None:
            for k in self.x1:
                x_mem += sys.getsizeof(k)
                x_mem += sys.getsizeof(self.x1[k])
                
                for v in self.x1[k]:
                    x_mem += sys.getsizeof(v)
                    x_mem += sys.getsizeof(self.x1[k][v])
            
            for k in self.x2:
                x_mem += sys.getsizeof(k)
                x_mem += sys.getsizeof(self.x2[k])
                
                for v in self.x2[k]:
                    x_mem += sys.getsizeof(v)
                    x_mem += sys.getsizeof(self.x2[k][v])
            
            for k in self.x3:
                x_mem += sys.getsizeof(k)
                x_mem += sys.getsizeof(self.x3[k])
                
                for v in self.x3[k]:
                    x_mem += sys.getsizeof(v)
                    x_mem += sys.getsizeof(self.x3[k][v])
            

        even_mem = sys.getsizeof(self.evenUp)
        even_mem += sys.getsizeof(self.evenDown)
        
        if self.evenDown is not None:
            for k in self.evenDown:
                even_mem += sys.getsizeof(k)
                even_mem += sys.getsizeof(self.evenDown[k])
                
                for v in self.evenDown[k]:
                    even_mem += sys.getsizeof(v)
                    even_mem += sys.getsizeof(self.evenDown[k][v])
        
            for k in self.evenUp:
                even_mem += sys.getsizeof(k)
                even_mem += sys.getsizeof(self.evenUp[k])
                
                for v in self.evenUp[k]:
                    even_mem += sys.getsizeof(v)
                    even_mem += sys.getsizeof(self.evenUp[k][v])
        
        simple_mem = sys.getsizeof(self.simpleOracle)
        
        if self.simpleOracle is not None:
            simple_mem = sum(self.simpleOracle.get_memory_usage())
            
        return (B_mem, p_mem, delta_mem, I_mem, D_mem, x_mem, even_mem, simple_mem)
        
    
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

def rnd_pow(x):
    if x == 0:
        return 0
    return 2**(math.ceil(math.log2(x))) 
        
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
        plt.plot(range(3,76), time_use, c=colors[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage of the preprocessing algorithm")
    plt.show()
    
# =============================================================================
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else sum(m) for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of the oracle")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[0] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of B")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[1] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of p")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[2] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of delta")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[3] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of I")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[4] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of D")
#     plt.show()
# 
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[5] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of x")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[6] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of even")
#     plt.show()
#     
#     for i, mem_use in enumerate(mem_uses):
#         plt.plot(range(3,76), [None if m == None else m[7] for m in mem_use], c=colors[i])
#     #plt.ylim(0, 1000000000)
#     plt.xlabel("k")
#     plt.ylabel("bytes")
#     plt.title("Memory usage of the Thorup-Zwick Oracle")
#     plt.show()
# =============================================================================


G = parse("input_roads.txt")
sample_pair_dists = dict()
appx_factors = []

mem_uses = []
query_time_uses = []
preprocessing_time_uses = []

for k in range(75, 76):
    print(k)
    O = Oracle(k)
    print('Oracle Initialised')
    O.init_simple_oracle(G, k)
    print('B, p, and delta Initialised')
    O.init_D_and_I(G)
    print('D and I Initialised')
    O.init_evens(G, k)
    print('evenDown and evenUp Initialised')
    O.init_x(G, k)
    print('x1, x2 and x3 Initialised')
    O.init_MN_oracle(G, k)
    
    sample_pairs = []

    preprocessing_time_uses.append(O.preprocessingTime)    
    mem_uses.append(O.get_memory_usage())

# =============================================================================
#     for _ in tqdm(range(1000)):
#         u, v = sample(G.get_nodes(), 2)
#         
#         sample_pairs.append((u,v))
#         
#         dists = get_min_dist(G, u)    
#         sample_pair_dists[(u,v)] = dists[v]
# =============================================================================
    
    start = time.time()

    for _ in tqdm(range(50000)):
        
        u, v = sample(G.get_nodes(), 2)
        
        approx, deltaMN_time = O.query(u,v)
        
        O.queryTime -= deltaMN_time
        
    O.queryTime += time.time() - start
 
    query_time_uses.append(O.queryTime)
    print(O.queryTime)
    
    del O