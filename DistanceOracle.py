import numpy as np
from queue import PriorityQueue
from random import sample
from collections import defaultdict
from itertools import combinations

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
    test = 0
    dists = dict()
    
    for g in graph.nodes:
        dists[g] = float('inf')
        
    dists[node] = 0
        
    queue = PriorityQueue()
    
    queue.put((0, node))
    
    test = 0
    
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
    
def preprocess(G, k = 2):
    
    nodes = G.get_nodes()
    n = len(nodes)
    
    A = list()
    A.append(set(nodes))
    
    for i in range(1, k):
        sample_size = np.random.binomial(n, n**(-1/(k)))
        A.append(set(sample(A[i-1], sample_size)))
    
    if len(A[k-1]) == 0:
        raise Exception('The k\'th set was empty. How unfortunate')
    
    p_dists = [dict() for i in range(k+1)]
    p = [dict() for i in range(k+1)]
    B = {v: set() for v in G.get_nodes()}
    delta = defaultdict(lambda: float('inf'))
    
    for v in G.get_nodes():
        delta[(v,v)] = 0
        p_dists[k][v] = float('inf')
    
    for i in range(k-1, -1, -1):
        p_dists_i, p_i = get_p(G, A[i])
        
        p_dists[i] = p_dists_i
        p[i] = p_i
        
        delta_i, B_i = get_bunches(G, p_dists, p[i], A[i], delta, B, i)
        
        delta |= delta_i
        B |= B_i
        
    return (delta, B, p)
        
            
            
def get_p(G, A_i):
        
    queue = PriorityQueue()
    seen = set()
    
    p_dists_i = dict()
    p_i = dict()
    
    for n in A_i:
        queue.put((0, n))
        p_dists_i[n] = 0
        p_i[n] = n
        
    while not queue.empty():
                
        cur_dist, cur_node = queue.get()
        
        for n in G.get_node(cur_node).edges:
            cost = G.get_node(cur_node).edges[n]
            
            if n.get_id() not in p_dists_i:
                p_dists_i[n.get_id()] = float('inf')
                    
            if cur_dist + cost < p_dists_i[n.get_id()]:
                p_dists_i[n.get_id()] = cur_dist + cost
                p_i[n.get_id()] = p_i[cur_node]
                
                if n.get_id() not in seen:
                    queue.put((p_dists_i[n.get_id()], n.get_id()))
                    seen.add(n.get_id())
                
    return (p_dists_i, p_i)
            
def get_bunches(G, p_dists, p_i, A_i, delta, B, i):
    
    queue = PriorityQueue()
    
    for n in A_i:
        queue.put((0, n))
        seen = set()
        
        while not queue.empty():
            cur_dist, cur_node = queue.get()
            
            for v in G.get_node(cur_node).get_neighbors():
                new_dist = delta[(n, cur_node)] + G.get_node(cur_node).edges[v]
                
                if new_dist < p_dists[i+1][v.get_id()]:
                    best = delta[(n, v.get_id())]
                    
                    if new_dist < best:
                        delta[(n, v.get_id())] = new_dist
                        B[v.get_id()].add(n)
                        
                        if v.get_id() not in seen:
                            queue.put((new_dist, v.get_id()))
                            seen.add(v.get_id())
    return (delta, B)


def query(B, delta, p, u, v):
    w = u
    i = 0
    
    while w not in B[v]:
        i += 1
        u, v = v, u
        w = p[i][u]
        
    return delta[(w,u)] + delta[(w, v)]
    


G = parse()
dists = get_min_dist(G, 'A')
delta, B, p = preprocess(G)

stretchSum = 0.0
for u, v in combinations(G.get_nodes(), 2):
    u_dists = get_min_dist(G, u)
    approx = query(B, delta, p, u, v)
    stretchSum += approx/u_dists[v]
    
    if approx/u_dists[v] > 3.0:
        print("oh no")
    
print(stretchSum / len(list(combinations(G.get_nodes(), 2))))
    
        