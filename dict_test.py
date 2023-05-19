import random
import time
from tqdm import tqdm

d1 = dict()
d2 = dict()

for i in range(100):
    d1[str(i)] = random.uniform(0, 75)
    
for i in range(1000000000):
    d2[str(i)] = random.uniform(0, 75)
    
l1 = []
l2 = []
  
for i in tqdm(range(5000000)):
    l1.append(d1[str(i % 100)])
    
for i in tqdm(range(5000000)):
    l2.append(d2[str(i % 10000000)])
    