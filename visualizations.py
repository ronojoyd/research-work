#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install simpy')
get_ipython().system('pip install networkx')


# In[236]:


import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import operator
from matplotlib import pyplot as plt
from scipy.spatial.distance import *

final_array = [[], [], [], []]

def place_node_in_zone(D, f):

    global lat_dist, lon_dist, rB
    return (random.uniform(D[f][0] - rB / lat_dist, D[f][0] + rB / lat_dist),
            random.uniform(D[f][1] - rB / lon_dist, D[f][1] + rB / lon_dist))


def mobile():
    # Borough coordinates
    BC = {'Manhattan': (40.7831, 73.9712),
          'Bronx': (40.8448, 73.8648),
          'Brooklyn': (40.6782, 73.9442),
          'Queens': (40.7282, 73.7949),
          'Staten Island': (40.5795, 74.1502)
          }

    # District coordinates
    DC = {'Manhattan': [(40.7163, 74.0086), (40.7336, 74.0027), (40.7150, 73.9843), (40.7465, 74.0014), (40.7549, 73.9840), (40.7571, 73.9719),
                       (40.7870, 73.9754), (40.7736, 73.9566), (40.8253, 73.9476), (40.8089, 73.9482), (40.7957, 73.9389), (40.8677, 73.9212)],

          'Bronx': [(40.8245, 73.9104), (40.8248, 73.8916), (40.8311, 73.9059), (40.8369, 73.9271), (40.8575, 73.9097), (40.8535, 73.8894),
                    (40.8810, 73.8785), (40.8834, 73.9051), (40.8303, 73.8507), (40.8398, 73.8465), (40.8631, 73.8616), (40.8976, 73.8669)],

          'Brooklyn': [(40.7081, 73.9571), (40.6961, 73.9845), (40.6783, 73.9108), (40.6958, 73.9171), (40.6591, 73.8759), (40.6734, 74.0083),
                       (40.6527, 74.0093), (40.6694, 73.9422), (40.6602, 73.9690), (40.6264, 74.0299), (40.6039, 74.0062), (40.6204, 73.9600),
                       (40.5755, 73.9707), (40.6415, 73.9594), (40.6069, 73.9480), (40.6783, 73.9108), (40.6482, 73.9300), (40.6233, 73.9322)],

          'Queens': [(40.7931, 73.8860), (40.7433, 73.9196), (40.7544, 73.8669), (40.7380, 73.8801), (40.7017, 73.8842), (40.7181, 73.8448),
                     (40.7864, 73.8390), (40.7136, 73.7965), (40.7057, 73.8272), (40.6764, 73.8125), (40.7578, 73.7834), (40.6895, 73.7644),
                     (40.7472, 73.7118), (40.6158, 73.8213)],

          'Staten Island': [(40.6323, 74.1651), (40.5890, 74.1915), (40.5434, 74.1976)]
          }

    return DC

def find_dist(x1, y1, x2, y2):

    global mTOm, lat_dist, lon_dist
    return math.sqrt(math.pow(x2 - x1, 2) * lat_dist + math.pow(y2 - y1, 2) * lon_dist) * mTOm


class Node(object):

    def __init__(self, env, ID, waypoints, my_coor):

        global T, periodicity

        self.ID = ID
        #print(self.ID)
        self.env = env

        # Neighbor list
        self.nlist = []

        self.old_coor = None

        self.my_coor = my_coor

        self.start = True

        self.recBuf = simpy.Store(env, capacity = recBufferCapacity)

        # Time instant
        self.ti = random.randint(0, PT - 1)

        # List of events detected by system
        self.buffer = []

        # List of events detected by system
        self.events = []

        if 'E-' in self.ID:
            self.rE = 1000.0
            self.f = None
            self.my_waypoint = np.random.choice([i for i in range(len(D))], size = periodicity)
            self.env.process(self.move())

        if self.ID == 'E-1':
            self.env.process(self.time_increment())

    def move(self):

        global Xlim, Ylim, D, a, G, step, baseE, final_array

        while True:
            if self.rE > baseE:

                self.f = self.my_waypoint[T % periodicity]
                self.my_coor = place_node_in_zone(D, self.f)
                
                if self.ID == 'E-0':
                    final_array[0].append(self.my_coor)
                elif self.ID == 'E-1':
                    final_array[1].append(self.my_coor)
                elif self.ID == 'E-2':
                    final_array[2].append(self.my_coor)
                elif self.ID == 'E-3':
                    final_array[3].append(self.my_coor)

            yield self.env.timeout(minimumWaitingTime)


    def time_increment(self):

        global T

        while True:
            T = T + 1
            #(self.my_coor, self.f, self.my_waypoint)
            yield self.env.timeout(minimumWaitingTime)


# Number of fog nodes
eG = 4

T = 0

# Move how often
mho = 3

# Base energy level
baseE = 10.0

# Unit for device memory
recBufferCapacity = 1000

# Simulation Duration
Duration = 9

# Pause time
PT = 5

# Fog sensing range
sensing_range = 500.0

# Mobile sensing range
sensing_range_mobile = 100.0

# Simulation range
Xlim = [40.0, 41.0]
Ylim = [73.0, 74.5]

# Define waypoints
how_many = 50
minimumWaitingTime = 1

E = mobile()
D = []
for p in E.keys():
    L = E[p]
    D.extend(L)

#print (D)
# Location of Base station
# BC = (np.sum(Xlim)/2, np.sum(Ylim)/2)
BC = np.mean(D, axis = 0)

# Proximity weighing factor
WF = 0.5

# Miles to meters
mTOm = 1609.34

# Probability of intra-zone mobility (for ORBIT)
pr = 0.9

# Promptness increment/decrement
prompt_incentive = 5

# Distance between two latitude and longitudes
lat_dist = 69.0
lon_dist = 54.6

# Weighing factor (for LATP)
aLATP = 1.2

# Area of NYC
A = 302.6

# Number of neighborhoods
nB = 10

# Periodicity
periodicity = 3

# radius of a neighborhood
AB = A/nB
rB = math.sqrt(AB/math.pi)

# Create Simpy environment and assign nodes to it.
env = simpy.Environment()

minimumWaitingTime = 1

# Number of mobile nodes
mG = 500

# Choice of waypoint
Coor = [int(i % 59) for i in range(eG + mG + 2)]

# Create Simpy environment and assign nodes to it.
env = simpy.Environment()
entities = []

for i in range(eG + mG + 2):

    if i < eG:
        # Edge device
        entities.append(Node(env, 'E-' + str(i), Coor, place_node_in_zone(D, Coor[i])))

    elif i == eG:
        # Event generator
        entities.append(Node(env, 'EG-' + str(i), None, None))

    else:
        # Base station
        entities.append(Node(env, 'BS-' + str(i), Coor, BC))

env.run(until = Duration)

# ------------------------------------------------------
#print(final_array)
 
plt.plot([final_array[0][alpha][1] for alpha in range(len(final_array[0]))], [final_array[0][alpha][0] for alpha in range(len(final_array[0]))], linestyle='--', marker='o', color='black')
plt.plot([final_array[1][alpha][1] for alpha in range(len(final_array[0]))], [final_array[1][alpha][0] for alpha in range(len(final_array[0]))], linestyle='--', marker='o', color='red')
#plt.plot([final_array[2][alpha][1] for alpha in range(len(final_array[0]))], [final_array[2][alpha][0] for alpha in range(len(final_array[0]))], linestyle='--', marker='o', color='blue')
plt.plot([final_array[3][alpha][1] for alpha in range(len(final_array[0]))], [final_array[3][alpha][0] for alpha in range(len(final_array[0]))], linestyle='--', marker='o', color='green')
plt.ylim(40.4, 41.0)
plt.xlim(73.6, 74.2)

plt.grid(linestyle='--')
plt.tight_layout()
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.tight_layout()
plt.savefig('nodes_movement.png', dpi=300)
plt.show()

print([final_array[0][alpha][1] for alpha in range(len(final_array[0]))])
print([final_array[1][alpha][1] for alpha in range(len(final_array[0]))])
print([final_array[3][alpha][1] for alpha in range(len(final_array[0]))])

print([final_array[0][alpha][0] for alpha in range(len(final_array[0]))])
print([final_array[1][alpha][0] for alpha in range(len(final_array[0]))])
print([final_array[3][alpha][0] for alpha in range(len(final_array[0]))])


# In[242]:


from matplotlib import pyplot as plt

plt.plot([73.83907993014601, 73.92334249854268, 73.8231444642337, 73.82266206709582, 73.83141069614301, 73.79374190307924, 73.86101224323824, 73.89531669258221, 73.88090944962997, 73.83907993014601], [40.79934804321979, 40.92129147844814, 40.73831856050903, 40.74335700025858, 40.85746747282539, 40.66302991823795, 40.78259764128123, 40.89769733221607, 40.6688778916615, 40.79934804321979], linestyle='--', marker='o', color='black')
plt.plot([73.96553557470297, 74.07591211856725, 73.6922197112931, 74.02571877183995, 73.99563215629222, 73.72841419893254, 73.96534592207688, 74.02520722756644, 73.67053999158216, 73.96553557470297], [40.770448268898534, 40.63991950860495, 40.75516571840618, 40.799344345627986, 40.600008584749055, 40.70476606063607, 40.766806152556086, 40.611543322385, 40.70826723072767, 40.770448268898534], linestyle='--', marker='o', color='red')
plt.plot([73.97847574485408, 74.00932255533847, 74.19977374054508, 73.95403888365236, 74.03774906564688, 74.16844276606741, 73.98679122224449, 73.9998165155389, 74.17492242537195, 73.97847574485408], [40.63443687268707, 40.778567705916046, 40.64775181639157, 40.68238468294029, 40.71488577253242, 40.62566065830048, 40.6860861395027, 40.72834609276257, 40.61916186250849, 40.63443687268707], linestyle='--', marker='o', color='green')
plt.ylim(40.5, 41.0)
plt.xlim(73.6, 74.28)

plt.grid(linestyle='--')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)


plt.tight_layout()
plt.savefig('nodes_movement_final.png', dpi=300)
plt.show()


# In[ ]:





# In[ ]:




