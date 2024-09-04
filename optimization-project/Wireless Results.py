#!/usr/bin/env python
# coding: utf-8

# In[5]:


import networkx as nx
import simpy
import numpy as np
#import pickle
import random
import math
import matplotlib.pyplot as plt
import operator

from itertools import permutations
from scipy.spatial.distance import *

arr1 = []
arr2 = []
arr3 = []


def eps_greedy_q_learning_with_table(q_table, period, periodicity, c_w, c_p, r, T, y = 0.95, lr = 0.8, decay_factor = 0.98):

    global indexes, eps, N

    my_index = (c_w, c_p)
    eps *= decay_factor
    i = indexes.index(my_index)
    last_period = (period - 1) % periodicity
    print(last_period, i, period)

    if N <= 3:
        q_table[last_period, i] = q_table[last_period, i] + lr * r
    else:
        q_table[last_period, i] = 2/(N + 1) * (lr * r) + (N - 1)/(N + 1) * q_table[last_period, i]

    if random.uniform(0, 1) < eps or np.sum(q_table[period, :]) == 0:
        a = random.randrange(0, len(indexes))
    else:
        a = np.argmax(q_table[period, :])

    print ('****Next step:', period, a)
    N = N + 1

    return q_table, indexes[a][0], indexes[a][1]


def find_dist(x1, y1, x2, y2):

    global mTOm, lat_dist, lon_dist
    return math.sqrt(math.pow(x2 - x1, 2) * lat_dist + math.pow(y2 - y1, 2) * lon_dist) * mTOm


def reward(t1, t2):

    global E_list, P_sensed, P_rec, Hcp

    P_s, P_r, L_s, L_r = [], [], [], []

    # Energy-reward
    e_r = np.mean([E_list[t] for t in range(t1, t2)])

    for t in range(t1, t2 + 1):
         P_r.extend(P_rec[t])

    for item in P_r:
        if item not in L_r:
            L_r.append(item)

    for t in range(t1, t2 + 1):
         P_s.extend(P_sensed[t])

    for item in P_s:
        if item not in L_s:
            L_s.append(item)

    print ([Hcp[item[0]] for item in P_r])
    if len([Hcp[item[0]] for item in P_r]) > 0:
        return len(L_r)/len(L_s), len(P_r)/(np.mean([Hcp[item[0]] for item in P_r]) + 1), 1.0/(e_r + 1.0)

    return len(L_r)/len(L_s), 0, 1.0/e_r


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


def avg_degree(G):

    H = G.to_undirected()
    return np.mean([H.degree(u) for u in H.nodes()])


def efficiency(G):

    global eG, perc
    num = 0
    den = 0

    perc = 0
    for u in G.nodes():
        for v in G.nodes():
            if v == u:
                continue
            if nx.has_path(G, u, v):
                num += 1.0/float(nx.shortest_path_length(G, u, v))
                if v == eG + 1:
                    perc += 1
            den += 1

    return num/den, perc


def find_directions(G):
    global BC, eG

    H = nx.DiGraph()
    H.add_nodes_from(list(G.nodes()))

    L = [eG + 1]
    while len(L) < len(H.nodes()):
        new_list = []
        for u in range(eG):
            if u in L:
                continue
            for v in L:
                if G.has_edge(u, v):
                    H.add_edge(u, v)

                new_list.append(u)

        for u in new_list:
            for v in new_list:
                if G.has_edge(u, v):
                    H.add_edge(u, v)
                    H.add_edge(v, u)

        L.extend(new_list)

    return H


class Node(object):

    def __init__(self, env, ID, waypoints, my_coor):

        global T, periodicity

        self.ID = ID
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

        self.NMC = {}

        if 'E-' in self.ID:
            self.rE = 2500.0
            self.f = None
            self.SD = {}
            self.next_hop = None

            self.my_waypoint = list(np.random.choice([i for i in range(len(D))], size = periodicity))
            self.env.process(self.move())
            self.env.process(self.sense())
            self.env.process(self.send())

        if self.ID == 'E-1':
            self.myG = nx.Graph()
            self.env.process(self.time_increment())

        if 'EG-' in self.ID:
            self.env.process(self.genEvent())

        if 'BS-' in self.ID:
            self.env.process(self.receive())

    def move(self):

        global Xlim, Ylim, D, a, G, step, baseE, periodicity

        while True:
            if T % mho == 0:
                if self.f is None:
                    self.f = self.my_waypoint[0]
                else:

                    # Periodic mobility
                    i = (self.my_waypoint.index(self.f) + 1) % periodicity
                    self.f = self.my_waypoint[i]

                    # Random mobility
                    # self.f = random.choice([i for i in range(len(D))])
                self.my_coor = place_node_in_zone(D, self.f)

            yield self.env.timeout(minimumWaitingTime)

    def time_increment(self):

        global T, eG, sensing_range, L, E_list, periodicity, RL, c_p, c_w, eps, base_neighbor, N, R, En, arr1, arr2

        while True:
            T = T + 1
            E_list[T] = np.std([entities[u].rE for u in range(eG)])/np.mean([entities[u].rE for u in range(eG)]) * 100000
            #arr1 = np.mean([entities[u].rE for u in range(eG)])/np.mean([entities[u].rE for u in range(eG)]) * 100000
            #arr2 = np.std([entities[u].rE for u in range(eG)])/np.mean([entities[u].rE for u in range(eG)]) * 100000
            
            if T % mho == 0:
                En.append(np.mean([entities[u].rE for u in range(eG)]))
                arr1.append(np.mean([entities[u].rE for u in range(eG)]))
                arr2.append(np.std([entities[u].rE for u in range(eG)]))
            # if T % mho == 1 and T > mho:
                # r = reward(T - mho, T)
                #
                # RL, c_w, c_p = eps_greedy_q_learning_with_table(RL, self.my_waypoint.index(self.f), periodicity, c_w, c_p, r[0], T)
                # # RL, c_w, c_p = eps_greedy_q_learning_with_table(RL, N % periodicity, periodicity, c_w, c_p, r[0], T)
                #
                # L = [entities[u].rE for u in range(eG)]
                # print ('Reward:', T, c_w, c_p, eps, len(list(entities[1].myG.edges())), base_neighbor, np.mean(L), np.min(L), R)
                # base_neighbor = [u for u in range(eG) if entities[1].myG.has_edge(u, eG + 1)]
                #
                # Trace.append((c_w, c_p))
                # # print (Trace)
                # print (RL)

            if T % frequencyEvent == 1:
                print ('********', T, self.f)

                nlist = [u for u in range(eG) if entities[u].rE > baseE]
                self.myG = nx.Graph()
                self.myG.add_node(eG + 1)
                self.myG.add_nodes_from(nlist)

                for u in nlist:
                    for v in nlist:
                        if u == v:
                            continue

                        if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], entities[v].my_coor[0], entities[v].my_coor[1]) <= sensing_range[c_p]:
                            self.myG.add_edge(u, v)

                    if find_dist(entities[u].my_coor[0], entities[u].my_coor[1], BC[0], BC[1]) <= sensing_range[c_p]:
                        self.myG.add_edge(u, eG + 1)

                self.myG = find_directions(self.myG)
                self.find_NMC()

                Ld.append(avg_degree(self.myG))

                e, _ = efficiency(self.myG)
                Le.append(e)

            # if T % mho == 1 or T % mho == 2:
            #     print (T, self.my_coor, self.f, entities[2].f,
            #     self.my_waypoint, len(self.myG.nodes()), len(self.myG.edges()), perc)
            #     print (T, self.my_coor, entities[0].SD, "\n")

            yield self.env.timeout(minimumWaitingTime)

    def genEvent(self):

        global T, Duration, frequencyEvent, globalEventCounter, Xlim, Ylim

        while True:

            # Generate new events
            if T % frequencyEvent == 0:
                #ev = events[T]
                for i in range(how_many_events):

                    # Random event
                    new_event = [globalEventCounter, (random.uniform(Xlim[0], Xlim[1]), random.uniform(Ylim[0], Ylim[1])), T]
                    # new_event = ev[i]
                    # self.events.append(new_event)
                    # globalEventCounter += 1

            # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
            self.events = self.updateEventList(self.events)

            yield self.env.timeout(minimumWaitingTime)

    def updateEventList(self, L):

        global frequencyEvent, T, recur

        remove_indices = []
        for each in L:
            if each[2] <= T - recur:
                remove_indices.append(L.index(each))

        return [i for j, i in enumerate(L) if j not in remove_indices]

    def sense(self):

        global baseE, L, sensing_range, eG, senseE, P_sensed, Hcp, c_p
        while self.rE > baseE:

            if T % frequencyEvent == 0:
                self.rE = self.rE - senseE[c_p]

                # Sense event in the vicinity
                for each in entities[eG].events:
                    if find_dist(each[1][0], each[1][1], self.my_coor[0], self.my_coor[1]) <= sensing_range[c_p]:
                        self.recBuf.put(each)
                        P_sensed[T].append(each)
                        Hcp[each[0]] = 0

                # Remove old events (i.e. which has been in the system for at least 'frequencyEvent' time) from list
                self.events = self.updateEventList(self.events)
                self.move()

            yield self.env.timeout(minimumWaitingTime)

    def send(self):

        global T, baseE, W, c_p, c_w

        while True:

            if 'E-' in self.ID and self.rE > baseE:

                # Filter out redundant event data and send to next hop
                L = []
                while len(self.recBuf.items) > 0:
                    item = yield self.recBuf.get()
                    if item not in L:
                        L.append(item)

                self.SD = {u: 0 for u in range(eG) if entities[1].myG.has_edge(int(self.ID[2:]), u) and entities[u].rE > baseE and nx.has_path(entities[1].myG, u, eG + 1)}

                # Send data to next gop or base station
                if len(list(self.SD.keys())) > 0:

                    L_s = {u: nx.shortest_path_length(entities[1].myG, u, eG + 1) for u in self.SD.keys() if entities[u].rE > baseE}
                    L_s = {u: (L_s[u] + 1)/(max(list(L_s.values())) + 1) for u in self.SD.keys() if entities[u].rE > baseE}

                    L_m = {u: entities[1].NMC[u] for u in self.SD.keys() if entities[u].rE > baseE}
                    L_m = {u: (L_m[u] + 1)/(max(list(L_m.values())) + 1) for u in self.SD.keys() if entities[u].rE > baseE}

                    self.SD = {u: W[c_w] * L_m[u] + (1.0 - W[c_w]) * L_s[u] for u in self.SD.keys()}

                    if find_dist(self.my_coor[0], self.my_coor[1], BC[0], BC[1]) <= sensing_range[c_p]:
                        # for item in L:
                        # entities[self.next_hop].recBuf.put(item)
                        # self.rE -= fg_fg_E
                        self.next_hop = eG + 1

                    else:
                        self.next_hop = max(self.SD, key = self.SD.get)

                    for item in L:
                        entities[self.next_hop].recBuf.put(item)
                        self.rE -= fg_fg_E
                        Hcp[item[0]] += 1

            yield self.env.timeout(minimumWaitingTime)

    def receive(self):

        global T, P_rec, R

        while True:

            while len(self.recBuf.items) > 0:
                item = yield self.recBuf.get()
                P_rec[T].append(item)
                R = R + 1

            # print ('***', T, len(P_rec[T]), np.mean([entities[u].rE for u in range(eG)]))

            yield self.env.timeout(minimumWaitingTime)

    def find_NMC(self):

        self.NMC = {u: 0.0 for u in self.myG.nodes()}
        self.NMC[int(self.ID[2:])] = 0

        for u in self.myG.nodes():
            for v in self.myG.nodes():
                if v <= u:
                    continue

                for w in self.myG.nodes():
                    if w <= v:
                        continue

                    if self.myG.has_edge(u, v) and self.myG.has_edge(v, w) and self.myG.has_edge(u, w):
                        self.NMC[u] += 1
                        self.NMC[v] += 1
                        self.NMC[w] += 1


Le = []
Ld = []
perc = None

eps = 0.7

# Motif vs. shortest_path trade-off
W = [0.2, 0.8]
c_w = 0

frequencyEvent = 2
globalEventCounter = 0

# Sense event data energy
senseE = [0.05, 0.10]

# How many events
how_many_events = 50

# How often should event stay in system
recur = 10

# Number of fog nodes
eG = 100

T = 0

N = 0

goBackInTime = 40

# Move how often
mho = 20

# Base energy level
baseE = 300.0

# Unit for device memory
recBufferCapacity = 1000

# Simulation Duration
Duration = 2000

# Pause time
PT = 5

# Fog sensing range
sensing_range = [400.0, 650.0]
c_p = 1

# Simulation range
Xlim = [40.5, 41.0]
Ylim = [73.7, 74.2]

# Define waypoints
how_many = 50
minimumWaitingTime = 1

E = mobile()
D = []
for p in E.keys():
    L = E[p]
    D.extend(L)

# print (D)
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
nB = 59

# Periodicity
periodicity = 3

base_neighbor = None

# Peer to FOG data transfer energy
fg_fg_E = 0.37

# Scan event data energy
scanE = 3.68

# radius of a neighborhood
AB = A/nB
rB = math.sqrt(AB/math.pi) * 0.05

# Create Simpy environment and assign nodes to it.
env = simpy.Environment()

minimumWaitingTime = 1

# Choice of waypoint
Coor = [int(i % 59) for i in range(eG + 2)]

# Create Simpy environment and assign nodes to it.
env = simpy.Environment()
entities = []

# Average residual energy of all node
E_list = {}

# List of packets sensed by any node
P_sensed = {t: [] for t in range(Duration + 10)}

# List of packets received by BS
P_rec = {t: [] for t in range(Duration + 10)}

# Hop count for packet
Hcp = {}

# RL Table
R = 0
# Weight X power level
indexes = [(i, j) for i in range(len(W)) for j in range(len(sensing_range))]
RL = np.zeros((periodicity, len(indexes)))
# print (RL)

Trace = []
En = []
#events = pickle.load(open('events.p', 'rb'))

for i in range(eG + 2):

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
print ('***', np.mean(Le), np.mean(Ld))
# print (Trace)
print ('Received:', R)

#pickle.dump(En, open('En-1.p', 'wb'))

# plt.plot([i for i in range(len(Trace))], [indexes.index((Trace[i][0], Trace[i][1])) for i in range(len(Trace))],
# #          linewidth = 2, color = 'green', label = 'power', marker = 'o')
# #
# #
# plt.xlabel('Time in minutes', fontsize = 15)
# plt.ylabel('RL Action', fontsize = 15)

# plt.tight_layout()
# plt.savefig('Energy.png', dpi = 300)
# plt.show()

C = (0, 0, 0, 0)
for v in [indexes.index((Trace[i][0], Trace[i][1])) for i in range(len(Trace))]:
    if v == 0:
        C = (C[0] + 1, C[1], C[2], C[3])
    elif v == 1:
        C = (C[0], C[1] + 1, C[2], C[3])
    elif v == 2:
        C = (C[0], C[1], C[2] + 1, C[3])
    else:
        C = (C[0], C[1], C[2], C[3] + 1)

print (C)

# E = {}
# T = 0
# while T < Duration:
#
#     if T % frequencyEvent == 0:
#         L = []
#         for i in range(how_many_events):
#             # Random event
#             new_event = [globalEventCounter, (random.uniform(Xlim[0], Xlim[1]), random.uniform(Ylim[0], Ylim[1])), T]
#             globalEventCounter += 1
#
#             L.append(new_event)
#         E[T] = L
#
#     T = T + 1
#
# print ({T: len(E[T]) for T in range(0, Duration, frequencyEvent)})
# pickle.dump(E, open('events.p', 'wb'))

print('arr1: ', arr1)
print('arr2: ', arr2)

#print('\nmean', np.mean(arr3, axis=0))
#print('std', np.std(arr3, axis=0))


# In[35]:


from matplotlib import pyplot as plt

#low_power = [2499.5489999999986, 2499.0489999999954, 2498.548999999995, 2498.0489999999922, 2497.5489999999913, 2497.0489999999886, 2496.548999999988, 2496.048999999985, 2495.5489999999845, 2495.048999999982, 2494.548999999981, 2494.048999999978, 2493.5489999999777, 2493.0489999999745, 2492.5489999999727, 2492.04899999997, 2491.5489999999695, 2491.0489999999663, 2490.548999999966, 2490.048999999963, 2489.5489999999622, 2489.0489999999595, 2488.548999999959, 2488.048999999956, 2487.5489999999554, 2487.0489999999527, 2486.548999999952, 2486.048999999949, 2485.5489999999486, 2485.0489999999454, 2484.5489999999436, 2484.048999999941, 2483.5489999999404, 2483.048999999937, 2482.5489999999368, 2482.048999999934, 2481.548999999933, 2481.0489999999304, 2480.54899999993, 2480.0489999999268, 2479.5489999999263, 2479.0489999999236, 2478.5489999999227, 2478.04899999992, 2477.5489999999195, 2477.0489999999163, 2476.5489999999145, 2476.0489999999118, 2475.5489999999113, 2475.048999999908, 2474.5489999999077, 2474.048999999905, 2473.548999999904, 2473.0489999999013, 2472.548999999901, 2472.0489999998977, 2471.548999999897, 2471.0489999998945, 2470.5489999998936, 2470.048999999891, 2469.5489999998904, 2469.048999999887, 2468.5489999998854, 2468.0489999998827, 2467.548999999882, 2467.048999999879, 2466.5489999998786, 2466.048999999876, 2465.548999999875, 2465.048999999872, 2464.5489999998717, 2464.0489999998686, 2463.548999999868, 2463.0489999998654, 2462.5489999998645, 2462.0489999998617, 2461.5489999998613, 2461.048999999858, 2460.5489999998563, 2460.0489999998535, 2459.548999999853, 2459.04899999985, 2458.5489999998495, 2458.0489999998467, 2457.548999999846, 2457.048999999843, 2456.5489999998426, 2456.0489999998395, 2455.548999999839, 2455.0489999998363, 2454.5489999998354, 2454.0489999998326, 2453.548999999832, 2453.048999999829, 2452.548999999827, 2452.0489999998244, 2451.548999999824, 2451.048999999821, 2450.5489999998204, 2450.0489999998176]
#low_power_std = [0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025462, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025466, 0.007000000000025468, 0.007000000000025466, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.00700000000002547, 0.007000000000025466, 0.007000000000025462]

low_power = [2499.5489999999986, 2498.930599999996, 2498.386199999995, 2497.6937999999923, 2496.808999999992, 2496.216499999989, 2495.6942999999883, 2494.9352999999855, 2494.368699999985, 2493.676299999982, 2492.8432999999814, 2492.265599999979, 2491.676799999978, 2490.7771999999754, 2490.2068999999738, 2489.5625999999706, 2488.6518999999703, 2488.0667999999673, 2487.3965999999673, 2486.667199999964, 2486.0931999999634, 2485.4377999999606, 2484.57889999996, 2484.0418999999574, 2483.3420999999566, 2482.501699999954, 2481.909199999953, 2481.0280999999504, 2480.31719999995, 2479.802399999947, 2479.0507999999454, 2478.214099999943, 2477.680799999942, 2477.0179999999395, 2476.358899999939, 2475.8107999999356, 2475.1035999999353, 2474.0226999999327, 2473.470899999932, 2472.771099999929, 2472.0823999999284, 2471.5749999999257, 2470.815999999925, 2470.186499999922, 2469.605099999922, 2468.801699999919, 2468.101899999917, 2467.5833999999145, 2466.779999999914, 2465.872999999911, 2465.3285999999107, 2464.584399999908, 2463.651499999907, 2463.103399999904, 2462.4331999999035, 2461.444799999901, 2460.867099999901, 2460.2486999998973, 2459.430499999897, 2458.8786999998943, 2458.201099999894, 2457.382899999891, 2456.834799999889, 2456.179399999886, 2455.187299999886, 2454.6650999998833, 2453.8098999998824, 2453.1433999998794, 2452.595299999879, 2451.954699999876, 2450.988499999876, 2450.421899999873, 2449.818299999872, 2448.9778999998694, 2448.4408999998686, 2447.670799999866, 2446.6564999998654, 2446.0824999998626, 2445.375299999861, 2444.564499999858, 2444.0274999998574, 2443.1722999998547, 2442.2541999998543, 2441.7060999998516, 2441.021099999851, 2440.010499999848, 2439.4475999998476, 2438.880999999845, 2437.9628999998445, 2437.3888999998417, 2436.689099999841, 2435.674799999838, 2435.148899999838, 2434.386199999835, 2433.542099999833, 2433.0161999998304, 2432.2941999998297, 2431.557399999827, 2430.9944999998265, 2430.2946999998235]
low_power_std = [0.007000000000025466, 0.6838725319821842, 0.9319579175045872, 1.6184515933442218, 3.2938219441848386, 3.769929541780802, 3.8341315196523085, 5.108063910131175, 5.474712532177587, 6.049768202995748, 7.6145724837291935, 8.035748418160058, 8.317262636225886, 10.244040324009763, 10.625426268622514, 11.070889902800392, 13.114081644930325, 13.581838673754417, 14.090268004544232, 15.15379867095589, 15.554767750109841, 16.056593261331866, 17.796651757839538, 18.000121315974752, 18.657144974239618, 20.30824748986866, 20.805718765756318, 22.127274874907464, 23.086767598771498, 23.160260193695652, 24.126781744767715, 25.640365211705003, 25.808063688692474, 26.449908128377064, 27.168097573984905, 27.382623931237084, 28.22323959150561, 30.845592322233635, 31.076894603377447, 31.864057867594347, 32.71151329791039, 32.74997720608928, 33.80949254276564, 34.38124931920454, 34.764908212007214, 36.055556993468194, 36.93504137521975, 37.02071842143519, 38.361599732012365, 40.10756702916609, 40.29379716580508, 41.37633370223841, 43.24393620785342, 43.46566344183515, 44.20894176248564, 46.34339705889861, 46.706542845609036, 47.211211966107015, 48.60891165978319, 48.856914856636116, 49.61110292856375, 50.9985094937927, 51.22445703917663, 51.882629825775155, 54.06288987566494, 54.15721805067418, 55.65572339830085, 56.382655404283014, 56.60562393356432, 57.209474852580755, 59.263016483705314, 59.58498469738752, 60.02131932995037, 61.53226138041544, 61.69528645032581, 62.82984664439085, 65.10451522550107, 65.45205795654917, 66.31635480714698, 67.69012387894777, 67.85356257962646, 69.3555799896963, 71.19371198608069, 71.4176514020658, 72.20174349547885, 74.46966656798409, 74.76452415575652, 75.04060620088752, 76.90903033185708, 77.2590827915169, 78.0781155009928, 80.38390830854229, 80.50945675997386, 81.58250318270444, 83.12976659768634, 83.25750548482569, 84.16708525519742, 85.2059094971443, 85.48334313037537, 86.30945331135744]

#high_power = [2499.098000000002, 2498.0980000000022, 2497.0980000000027, 2496.0980000000036, 2495.098000000005, 2494.098000000006, 2493.0980000000063, 2492.098000000007, 2491.0980000000086, 2490.098000000009, 2489.09800000001, 2488.0980000000104, 2487.0980000000122, 2486.0980000000127, 2485.098000000013, 2484.098000000014, 2483.0980000000154, 2482.0980000000163, 2481.098000000017, 2480.0980000000172, 2479.098000000019, 2478.0980000000195, 2477.0980000000213, 2476.098000000022, 2475.0980000000236, 2474.0980000000245, 2473.098000000025, 2472.0980000000254, 2471.0980000000272, 2470.0980000000277, 2469.0980000000286, 2468.098000000029, 2467.098000000031, 2466.0980000000313, 2465.098000000032, 2464.0980000000327, 2463.098000000034, 2462.098000000035, 2461.0980000000354, 2460.098000000036, 2459.0980000000377, 2458.098000000038, 2457.098000000039, 2456.0980000000395, 2455.0980000000413, 2454.098000000042, 2453.0980000000422, 2452.098000000043, 2451.0980000000445, 2450.0980000000454, 2449.098000000046, 2448.0980000000463, 2447.098000000048, 2446.0980000000486, 2445.0980000000504, 2444.098000000051, 2443.0980000000527, 2442.0980000000536, 2441.098000000054, 2440.0980000000545, 2439.0980000000563, 2438.098000000057, 2437.0980000000577, 2436.098000000058, 2435.09800000006, 2434.0980000000604, 2433.098000000061, 2432.098000000062, 2431.098000000063, 2430.098000000064, 2429.0980000000645, 2428.098000000065, 2427.098000000067, 2426.0980000000673, 2425.098000000068, 2424.0980000000686, 2423.0980000000704, 2422.098000000071, 2421.0980000000714, 2420.0980000000723, 2419.0980000000736, 2418.0980000000745, 2417.098000000075, 2416.0980000000754, 2415.0980000000773, 2414.0980000000777, 2413.0980000000795, 2412.09800000008, 2411.098000000082, 2410.0980000000827, 2409.098000000083, 2408.0980000000836, 2407.0980000000854, 2406.098000000086, 2405.098000000087, 2404.0980000000873, 2403.098000000089, 2402.0980000000895, 2401.09800000009, 2400.098000000091]
#high_power_std = [0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987266, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987271, 0.013999999999987271, 0.013999999999987266, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987267, 0.013999999999987271, 0.013999999999987262, 0.013999999999987267, 0.013999999999987271, 0.013999999999987271]

high_power = [2473.5976000000082, 2417.5934000000257, 2347.773400000047, 2255.820000000073, 2168.6439000000955, 2066.0989000001205, 1942.4862000001476, 1819.326300000174, 1687.4929000001994, 1572.5801000002205, 1515.3599000002368, 1430.4186000002549, 1355.005900000272, 1265.8091000002858, 1149.5800000003096, 1034.6089000003294, 963.8448000003399, 882.3720000003533, 783.2440000003704, 702.4237000003774, 641.6543000003886, 595.6901000003941, 569.1786000003981, 543.6690000004018, 502.37850000040794, 485.8017000004111, 469.783000000414, 451.7466000004167, 438.05930000041934, 424.93810000042043, 406.74620000042216, 398.8923000004231, 388.31610000042417, 381.0149000004249, 373.242700000426, 364.78430000042715, 358.31840000042774, 353.5234000004283, 346.7341000004293, 342.99690000042955, 341.10900000042983, 337.3156000004303, 333.7701000004306, 329.0850000004315, 325.43220000043215, 322.0199000004323, 317.45320000043233, 314.75130000043237, 311.0356000004325, 308.92200000043255, 305.71690000043253, 302.58740000043264, 296.41750000043265, 294.85640000043276, 293.07690000043283, 291.43730000043286, 290.414500000433, 289.5508000004331, 286.05270000043316, 285.04840000043333, 284.2069000004333, 282.62560000043334, 281.78560000043336, 281.5894000004334, 279.9428000004334, 279.75660000043337, 279.66660000043345, 279.5766000004335, 279.4866000004335, 278.7898000004336, 278.0486000004337, 277.8772000004337, 277.30620000043376, 277.12370000043376, 277.03370000043384, 276.94370000043386, 276.8537000004339, 276.7637000004339, 276.6737000004339, 275.84740000043394, 275.67970000043397, 274.8719000004341, 273.7654000004341, 273.6854000004342, 273.60540000043414, 273.5254000004342, 273.44540000043423, 272.9436000004343, 272.75630000043435, 272.67630000043437, 272.1967000004345, 272.0057000004345, 271.9257000004346, 271.8457000004346, 271.7657000004346, 271.6857000004347, 271.60570000043464, 271.5257000004347, 271.4457000004347, 271.0068000004348]
high_power_std = [23.936483497783502, 102.04687878829935, 170.39827010396223, 290.01266382341953, 418.0015078270744, 503.8034836656859, 593.9349564753905, 673.1325673261895, 753.994925804144, 811.1428181798693, 797.6834147773092, 806.742681345072, 803.3053142722205, 822.6911711681283, 807.7445212106416, 798.7173653299446, 789.3716292233883, 768.3941833784537, 729.2518248724931, 717.2616378318143, 667.5263713325502, 635.8962051647225, 612.4682259366234, 589.553028031342, 550.7775040782716, 533.7307980040666, 516.9720038463919, 495.69463042320666, 475.56622445719967, 461.1979378112594, 443.998061578574, 433.17826502686347, 422.31846172025905, 414.23694788244967, 402.0326239992097, 387.5470288397184, 378.75665551040134, 371.8815834461635, 358.34890458347826, 353.16103536967273, 349.36138176534314, 341.4883145272049, 336.6060713341066, 327.2706691211311, 320.56055181377803, 315.7241087864282, 307.99800931459635, 303.8945307196303, 299.7473468850635, 296.8566156614918, 293.31542368819396, 288.93127020666185, 281.8732948804211, 279.68235232319375, 277.21713171697263, 275.56937061964163, 273.5306936647982, 271.3507058316894, 268.9424718591432, 267.5937742987236, 266.56373054372256, 265.1246395464525, 263.60806461987534, 263.2908371889087, 261.9534696967316, 261.6528485654925, 261.4248424508211, 261.19695085976156, 260.9691740923375, 258.8837082204229, 258.07341315222135, 257.795263118152, 256.13017738555817, 255.59014995752008, 255.36611310294808, 255.14220052219454, 254.91841254273453, 254.69474949301153, 254.47121170243977, 253.87442356259126, 253.6136492559264, 251.36358635925845, 250.59271825980596, 250.37740494868652, 250.1622005276549, 249.94710527797412, 249.73211948173338, 248.34929739590356, 247.790584573565, 247.57658413369614, 246.34036180477878, 245.7981240215422, 245.58519923747338, 245.3723896376884, 245.1596955221425, 244.94711719166992, 244.73465494798597, 244.5223090936895, 244.31007993226538, 243.1297637307289]

plt.plot([i for i in range(len(low_power))], low_power, label='Low Power', color='red')
plt.plot([i for i in range(len(low_power))], [low_power[i]+low_power_std[i] for i in range(len(low_power))], linestyle='--', color='red')
plt.plot([i for i in range(len(low_power))], [low_power[i]-low_power_std[i] for i in range(len(low_power))], linestyle='--', color='red')
plt.fill_between([i for i in range(len(low_power))], [low_power[i]+low_power_std[i] for i in range(len(low_power))], [low_power[i]-low_power_std[i] for i in range(len(low_power))], color='#e67777', alpha=0.35)


plt.plot([i for i in range(len(high_power))], high_power, label='High Power', color='green')
plt.plot([i for i in range(len(high_power))], [high_power[i]+high_power_std[i] for i in range(len(high_power))], linestyle='--', color='green')
plt.plot([i for i in range(len(high_power))], [high_power[i]-high_power_std[i] for i in range(len(high_power))], linestyle='--', color='green')
plt.fill_between([i for i in range(len(high_power))], [high_power[i]+high_power_std[i] for i in range(len(high_power))], [high_power[i]-high_power_std[i] for i in range(len(high_power))], color='#88eb86', alpha=0.35)


plt.xlabel('Time in Minutes', fontsize=14)
plt.ylabel('Mean Residual Energy in Joules', color='black', fontsize=14)
plt.legend(prop={'size': 15})
plt.tight_layout()
plt.savefig('En-Res.png', dpi = 300)
plt.show()


# In[ ]:




