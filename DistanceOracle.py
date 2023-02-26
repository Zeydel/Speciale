import numpy as np
import heapq as heap
from queue import PriorityQueue
from random import sample
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

# Preprocess a graph into a Distance Oracle according to the algorithm
# described by Thorup and Zwick
def preprocess(G, k = 2):
    
    # Get the nodes from the graph
    nodes = G.get_nodes()
    n = len(nodes)
  
    # First, create the subsets of decreasing size
    A = list()
    
    # A[0] contains every vertex in the graph
    A.append(set(nodes))
    
    # A[1] to A[k-1] contains every element of the previous set with
    # probability n^(-1/k)
    for i in range(1, k):
        # Use a binomial distribution to determine the size of the sampling
        sample_size = np.random.binomial(len(A[i-1]), len(A[i-1])**(-1/(k)))
        
        # Use builtin method to pick random vertices from previous sets
        A.append(set(sample(A[i-1], sample_size)))
    
    # A[k] is the empty set
    A.append(set())
    
    # If we have zero nodes in the k-1'th set, abort
    if len(A[k-1]) == 0:
        raise Exception('The k\'th set was empty. How unfortunate')
    
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
    return (delta, B, p)
        
            
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


# Feeling cute, might delete later
def get_bunches(G, delta_Ai, A, delta, B, i):
    
    queue = []
    
    for n in tqdm(A[i] - A[i+1]):
        queue.put((0, n))
        seen = set()
        
        while not queue.empty():
            cur_dist, cur_node = queue.get()
            
            for v in G.get_node(cur_node).get_neighbors():
                new_dist = delta[(n, cur_node)] + G.get_node(cur_node).edges[v]
                
                if new_dist < delta_Ai[i+1][v.get_id()]:
                    best = delta[(n, v.get_id())]
                    
                    if new_dist < best:
                        delta[(n, v.get_id())] = new_dist
                        B[v.get_id()].add(n)
                        
                        if v.get_id() not in seen:
                            queue.put((new_dist, v.get_id()))
    return (delta, B)

# Function to get clusters for every node in a layer
def get_clusters(G, A, C, delta, delta_Ai, i):
    
    # Init queue
    queue = []
    
    # For every sampled node, not in the next layer
    for w in tqdm(A[i] - A[i+1]):
        
        
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
k = 2
delta, B, p = preprocess(G, k)

stretchSum = 0.0

actual_dists = dict()

query(B, delta, p, 'S', 'IN')

for _ in tqdm(range(1000)):
    
    u, v = sample(set(G.get_nodes()), 2)
    
    if u not in actual_dists:
        actual_dists[u] = get_min_dist(G, u)
    
    approx = query(B, delta, p, u, v)
    stretchSum += approx/actual_dists[u][v]
    
    if approx/actual_dists[u][v] > (2*k)-1:
        print(u)
        print(v)
        print(approx/actual_dists[u][v])
    
print(stretchSum / len(list(combinations(G.get_nodes(), 2))))
    
        