import random
import math
import os
from tqdm import tqdm


def generateId(last=''):
    
    if last == '':
        return 'A'
    
    if ord(last[-1]) < ord('Z'):
        return last[:-1] + chr(ord(last[-1])+1)
    
    return generateId(last[:-1]) + 'A'

def getDistance(src, dst):
    
    return math.sqrt((src[0]-dst[0])**2 + (src[1]-dst[1])**2 + (src[2]-dst[2])**2)

def generateInput(num_nodes, size, seed=42, connectivity=0.5):
    
    f = open("input.txt", 'a')
    
    random.seed(seed)
    
    nodes = set()
    
    last = ''
    
    for n in tqdm(range(num_nodes)):
        
        x = random.randint(0, size)
        y = random.randint(0, size)
        z = random.randint(0, size)
        node_id = generateId(last)
        last = node_id
        
        node = (node_id, x, y, z)
        
        string = node[0]
        
        for dst in nodes:
            
            if random.random() <= connectivity:
                
                string += ' ' + dst[0] + ':' + str(round(getDistance(node[1:], dst[1:])))
                
        f.write(string + '\n')
        
        nodes.add(node)
        
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
        
        
        
if os.path.exists("input.txt"):
    os.remove("input.txt")
generateLineGraph(1000000)