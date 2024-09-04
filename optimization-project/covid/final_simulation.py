#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pulp
import os
import random
import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import operator
import itertools
import pandas as pd
import decimal
import matplotlib.pyplot as plt
from geopy import distance
from scipy.spatial.distance import *
from scipy.spatial.distance import *
from scipy.optimize import minimize, differential_evolution
from scipy import optimize

def latp(pt, W, a):

    # Available locations
    AL = [k for k in W.keys() if euclidean(W[k], pt) > 0]
    AL = cutoff(AL, W, pt)

    if len(AL) == 0:
        return pt

    den = np.sum([1.0 / math.pow(float(euclidean(W[k], pt)), a) for k in sorted(AL)])

    plist = [(1.0 / math.pow(float(euclidean(W[k], pt)), a) / den) for k in sorted(AL)]

    next_stop = np.random.choice([k for k in sorted(AL)], p = plist, size = 1)

    return W[next_stop[0]]

class Node(object):

    def __init__(self, env, ID, coor, state, alpha, beta, gamma, delta, sigma, zone):

        global T, d

        self.ID = ID
        self.env = env

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.zone = zone
        
        # Neighbor list
        self.nlist = []

        self.old_coor = None
        self.new_coor = coor
        self.start = True

        self.ti = 3

        self.state = state

        if self.ID == 1:
            self.env.process(self.time_increment())
            self.env.process(self.optimizer())
            d = []

        self.env.process(self.move())
        #self.env.process(self.scan_neighbors())
        self.env.process(self.influence())

    def move(self):

        global Xlim, Ylim, W, a, zone_coordinates, R

        while True:

            # if T % mho == 0 and self.state != 'D':
            if T % mho == 0:
                #self.new_coor = (random.uniform(0, Xlim), random.uniform(0, Ylim))
                #print (self.ID, self.new_coor)
                
                c = zone_coordinates[self.zone]
                r = R[self.zone]

                # Define a set of k random points (potential next positions) within the circle of my current zone
                k = 10 
                P = []

                # Calculate the distance between current location (p) and each potential next hop
                D = []
                
                for i in range(k):
                    x = random.uniform(c[0] - r, c[0] + r)
                    y = random.uniform(c[1] - r, c[1] + r)
                    P.append((x, y))
                    D.append(euclidean(self.new_coor, (x, y)))
            
                # Select the next destination from P preferring short distances over long distances
                likelihood_of_selecting = [1.0/D[i] for i in range(k)]
                likelihood_of_selecting = [likelihood_of_selecting[i]/np.sum(likelihood_of_selecting) for i in range(k)]
                ind = np.random.choice([i for i in range(k)], p = likelihood_of_selecting, size = 1)[0]
               
                # New position of current agent
                self.new_coor = P[ind]
                #print(self.ID, self.new_coor)
                
            yield self.env.timeout(minimumWaitingTime)

    def scan_neighbors(self):

        global eG, sensing_range, entities, Coor

        while True:

            if T % PT == 2:

                self.nlist = []

                if self.start:
                    for u in range(eG):
                        if euclidean(self.new_coor, Coor[u]) <= sensing_range:
                            self.nlist.append(u)
                    self.start = False

                else:
                    for u in range(eG):
                        if euclidean(self.new_coor, entities[u].new_coor) <= sensing_range:
                            self.nlist.append(u)

                self.nlist = [u for u in self.nlist if u != self.ID]

            yield self.env.timeout(minimumWaitingTime)

    def influence(self):

        global minimumWaitingTime

        while True:
            if T % PT == (self.ti + 1) % PT:

                state_change = False

                if self.state == 'S':
                    for u in self.nlist:
                        if entities[u].state == 'E' and random.uniform(0, 1) <= self.alpha:
                            self.state = 'E'
                            state_change = True
                            break

                if self.state == 'S' and state_change == False:
                    for u in self.nlist:
                        if entities[u].state == 'I' and random.uniform(0, 1) <= self.beta:
                            self.state = 'I'
                            state_change = True
                            break

                if self.state == 'E' and state_change == False:
                    if random.uniform(0, 1) <= self.gamma:
                        self.state = 'I'
                        state_change = True

                '''
                if self.state == 'E' and state_change == False:
                    if random.uniform(0, 1) <= pi:
                        self.state = 'R'
                        state_change = True
                '''

                if self.state == 'I' and state_change == False:
                    if random.uniform(0, 1) <= self.delta:
                        self.state = 'R'
                        state_change = True

                if self.state == 'I' and state_change == False:
                    if random.uniform(0, 1) <= self.sigma:
                        self.state = 'D'
                        state_change = True

            yield self.env.timeout(minimumWaitingTime)

    def time_increment(self):

        global T, Tracker, D, sus, exp, inf, rec, dth

        while True:

            T = T + 1
            sus = len([i for i in range(eG) if entities[i].state == 'S']) 
            exp = len([i for i in range(eG) if entities[i].state == 'E']) 
            inf = len([i for i in range(eG) if entities[i].state == 'I']) 
            rec = len([i for i in range(eG) if entities[i].state == 'R'])
            dth = len([i for i in range(eG) if entities[i].state == 'D'])
            
            #print('sus: ' + str(sus) + ', exp: ' + str(exp) + ', inf: ' + str(inf) + ', rec: ' + str(rec) + ', dth: ' + str(dth))
            
            d.append((inf, rec, dth))

            # print (self.new_coor)
            if T % mho == 0 and self.old_coor != None:
                plt.scatter(self.new_coor[0], self.new_coor[1], s = 10, c = 'green')
                plt.plot([self.old_coor[   0], self.new_coor[0]], [self.old_coor[1], self.new_coor[1]], linestyle='dotted')

            yield self.env.timeout(minimumWaitingTime)

    def optimizer(self):
        
        global I, E, S, z, r, T, vaccines
        
        while True:
            if T % vaccine_interval == 0:
                    vaccines_per_zone = resource_allocation()
                    
                    arr_infected = np.zeros(z)
                    arr_suspected = np.zeros(z)
                    arr_exposed = np.zeros(z)
                    r = np.array([0.2 for y in range(z)])
                    
                    for chi in range(z):
                        arr_infected[chi] = len([i for i in range(eG) if entities[i].state == 'I' and entities[i].zone == chi])
                        arr_suspected[chi] = len([i for i in range(eG) if entities[i].state == 'S' and entities[i].zone == chi])
                        arr_exposed[chi] = len([i for i in range(eG) if entities[i].state == 'E' and entities[i].zone == chi])
                    
                    for delta in range(z):
                        available_vaccine = vaccines_per_zone[delta]
                        arr = [iota for iota in range(len(agent_zones)) if ((agent_zones[iota]==delta and entities[iota].state != 'D') and (entities[iota].state != 'R' and entities[iota].state != 'I'))]
                        vaccinated = []
                        immune = []
                        for phi in range(len(arr)):
                            while(available_vaccine > 0):
                                initial_state = entities[arr[phi]].state
                                entities[arr[phi]].state = np.random.choice([initial_state, 'R'], size=1, p=[1-r[delta], r[delta]])[0]
                                vaccinated.append(arr[phi])
                                if(entities[arr[phi]].state == "R"):
                                    immune.append(arr[phi])
                                    if initial_state == 'S':
                                        arr_suspected[delta] -= 1
                                    elif initial_state == 'E':
                                        arr_exposed[delta] -= 1
                                available_vaccine -= 1
                                break
                        
                        if(len(vaccinated) != 0):
                            r[delta] = expectedR + (((len(immune)/len(vaccinated)) - expectedR) * learning_rate)
       
                    I = np.array(arr_infected)
                    S = np.array(arr_suspected)
                    E = np.array(arr_exposed)
                    
            yield self.env.timeout(minimumWaitingTime)

            
            
# ----------------------------------------------------------------------------------

# Global current time
T = 0

def def_zone_and_coor():
    pop = np.array(file['Population'].values)
    pop = np.true_divide(pop, np.sum(pop))
    
    C = []
    coordinates = np.array(file['Location'].values)
    for a in range(0, len(coordinates)):
        arr = coordinates[a].split(', ')
        for b in range(0, len(arr)):
            arr[b] = float(arr[b])
        C.append((arr[0], arr[1]))
    agent_zones = np.random.choice(len(C), size=eG, p=pop)
    agent_coordinates = [C[d] for d in agent_zones]
    
    return C, agent_zones, agent_coordinates

def radius():
    area = np.true_divide(np.array(file['Population'].values), np.array(file['Population Density'].values))
    rad = [math.sqrt(area[i]/math.pi) for i in range(len(area))] / (np.max(np.array(file['Population'].values))/eG)
    return rad
          
def initial_state():
    
    global infected_ratio, susceptible_ratio, exposed_ratio
    
    infected_ratio = np.true_divide(np.array(file['Total Infected'].values), np.array(file['Population'].values)) + infected_bias
    exposed_ratio = pe * (1 - infected_ratio)
    susceptible_ratio = 1 - (infected_ratio + exposed_ratio)
    state = [(np.random.choice(['S', 'E', 'I'], size=1, 
                               p=[susceptible_ratio[(agent_zones[c])],
                                  exposed_ratio[(agent_zones[c])],
                                  infected_ratio[(agent_zones[c])]])[0]) for c in range(len(agent_zones))]
    return state

def resource_allocation(time=T, T=vaccines, p=0.3, fl=0.05, fh=0.2):
    
    how_many = warehouse

    # Coordinate of each zone
    C = zone_coordinates
    
    # List of warehouses
    # LW = np.random.choice(list(C.keys()), size = how_many, replace = False)
    
    array = np.zeros((z, z))
    for row in range(z):
        for column in range(0, z):
            array[row][column] = euclidean(C[row], C[column])
    
    avg_distance = (np.mean(array, axis=1)).reshape(z, ) # average distance from warehouses

    LW = np.array([3, 7, 8, 5, 1])
    
    VW = {}
    current_warehouse = 0
    for f in range(1, T + 1):
        VW[f - 1] = LW[current_warehouse]
        if (f % (T / warehouse) == 0):
            current_warehouse += 1
    
    global B, N, I, E, S, r
            
    B = np.array(file['Population Density'].values) * p  # rate of disease spread
    N = np.array(num_agent_per_zone) # total population for each zone
    if time <= vaccine_interval:
        r = np.array([0.2 for y in range(z)])
        I = np.array([infected_ratio[t]*num_agent_per_zone[t] for t in range(len(N))])  # total number infected in each zone
        E = (N-I) * pe
        S = N - (I + E)  # susceptible in each zone
    
    # Instantiate our problem class
    model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)

    X = pulp.LpVariable.dicts("X", ((i, j) for i in range(T) for j in range(len(B))), lowBound=0, upBound=1.0, cat='Continuous')

    # Objective functions
    den_infected = float(np.max(S) * np.max(B)*z)
    model += np.sum([(B[b] * (I[b]/N[b]) * (S[b] - r[b] * pulp.lpSum([X[(j, b)] for j in range(T)]))) for b in range(z)])/den_infected  # infected

    den_economic = float(T * np.max(array))
    model += np.sum([X[j, b] * euclidean(C[VW[j]], C[b]) for j in range(T) for b in range(z)])/den_economic  # economic
    
    '''
    den_3 = float(np.max(I)*z)
    model += np.sum((I[b] - (r[b] * pulp.lpSum([X[(j, b)] for j in range(T)]))) for b in range(z))/den_3
    '''
    
    # Constraint 1
    for i in range(T):
        model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) == 1

    # Constraint 2
    s = 0.0
    for i in range(T):
        s += pulp.lpSum([X[(i, j)] for j in range(len(B))])

    model += s == T

    # Constraint 3 (fairness upper)
    for i in range(len(B)):
        model += pulp.lpSum([X[(j, i)] for j in range(T)]) >= fl * T

    # Constraint 4 (fairness lower)
    for i in range(len(B)):
        model += pulp.lpSum([X[(j, i)] for j in range(T)]) <= fh * T

    model.solve()
    #print(pulp.LpStatus[model.status])
    #print ("With economic factor: " + str(pulp.value(model.objective)))
    
    # Transferred the pulp decision to the numpy array (A)
    A = np.zeros((T, len(B)))
    for i in range(T):
        for j in range(len(B)):
            A[i, j] = X[(i, j)].varValue

    vaccines_per_zone = []
    for i in range(len(B)):
        vaccines_per_zone.append(np.sum(A[:, i]))
        
    return np.array(vaccines_per_zone)
    


# Create Simpy environment and assign nodes to it. ---------------------------------------
env = simpy.Environment()

# Number of agents deployed in the simulation
eG = 300

# Number of zones
z = 10

# Number of vaccines being administered
vaccines = 100

# Number of warehouses
warehouse = 5

# File used for importing data
filename='covid-confirmed-ny-updated/covid_confirmed_NY_updated'
file = pd.read_csv("/kaggle/input/" + filename + ".csv")
file = file.iloc[0:z,:]

# Initial position and coordinates of node based on population density likelihood PW 
zone_coordinates, agent_zones, Coor = def_zone_and_coor()

# Number of agents in each zone
num_agent_per_zone = [0 for p in range(z)]
for epsilon in range(len(agent_zones)):
    num_agent_per_zone[agent_zones[epsilon]] += 1

# Fraction of susceptible/exposed nodes
pe = 0.3

# Simulation area --> not relevant right now
Xlim, Ylim = 100, 100

# Move how often
mho = 3

# Time intervals for administering vaccines
vaccine_interval = 2

# Minimum waiting time
minimumWaitingTime = 1

# Simulation duration
Duration = 3

# Susceptible-> Exposed
alpha = 0.2

# Susceptible-> Infected
beta = 0.2

# Exposed-> Infected
gamma = 0.2

# Infected-> Recovered
delta = 0.2

PT = 10

# Infected-> Death
sigma = 0.2

sensing_range = 30

# Scaled radius of each zone
R = radius()

# Learning rate for r
expectedR = 0.2
learning_rate = 0.1

# Variable used to increase proportion of infected (in case it is too low)
infected_bias = 0.10 

# List of node initial states
STATE = initial_state()

entities = [Node(env, i, Coor[i], STATE[i], alpha, beta, gamma, delta, sigma, agent_zones[i]) for i in range(eG)]
env.run(until = Duration)

