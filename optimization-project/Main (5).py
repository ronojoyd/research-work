import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import operator

from itertools import permutations
from scipy.spatial.distance import *


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

        global T, eG, sensing_range, L, E_list, periodicity, RL, c_p, c_w, eps, base_neighbor, N, R, En

        while True:
            T = T + 1
            E_list[T] = np.std([entities[u].rE for u in range(eG)])/np.mean([entities[u].rE for u in range(eG)]) * 100000

            if T % mho == 0:
                En.append(np.mean([entities[u].rE for u in range(eG)]))
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
                ev = events[T]
                for i in range(how_many_events):

                    # Random event
                    # new_event = [globalEventCounter, (random.uniform(Xlim[0], Xlim[1]),
                    # random.uniform(Ylim[0], Ylim[1])), T]

                    new_event = ev[i]
                    self.events.append(new_event)
                    globalEventCounter += 1

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
events = pickle.load(open('events.p', 'rb'))

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

pickle.dump(En, open('En-1.p', 'wb'))

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
