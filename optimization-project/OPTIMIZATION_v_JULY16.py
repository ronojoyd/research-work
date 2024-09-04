#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pulp
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
from geopy.distance import geodesic
from scipy.spatial.distance import *
from scipy.optimize import minimize, differential_evolution
from scipy import optimize
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.metrics import silhouette_score, calinski_harabasz_score

learning_rate_change = []
double_arr = []

for iteration in range(1): 

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
            self.env.process(self.influence())

        def move(self):

            global Xlim, Ylim, W, a, zone_coordinates, R

            while True:

                # if T % mho == 0 and self.state != 'D':
                if T % mho == 0:

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
                    '''
                    if self.state == 'S' and state_change == False:
                        for u in self.nlist:
                            if entities[u].state == 'I' and random.uniform(0, 1) <= self.beta:
                                self.state = 'I'
                                state_change = True
                                break
                    '''

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

            global Tracker, T, D, sus, exp, inf, rec, dth

            while True:

                T = T + 1
                sus = len([i for i in range(eG) if entities[i].state == 'S']) 
                exp = len([i for i in range(eG) if entities[i].state == 'E']) 
                inf = len([i for i in range(eG) if entities[i].state == 'I']) 
                rec = len([i for i in range(eG) if entities[i].state == 'R'])
                dth = len([i for i in range(eG) if entities[i].state == 'D'])

                # print('sus: ' + str(sus) + ', exp: ' + str(exp) + ', inf: ' + str(inf) + ', rec: ' + str(rec) + ', dth: ' + str(dth))

                d.append((inf, rec, dth))

                # print (self.new_coor)
                if T % mho == 0 and self.old_coor != None:
                    plt.scatter(self.new_coor[0], self.new_coor[1], s = 10, c = 'green')
                    plt.plot([self.old_coor[   0], self.new_coor[0]], [self.old_coor[1], self.new_coor[1]], linestyle='dotted')

                yield self.env.timeout(minimumWaitingTime)

        def optimizer(self):

            global I, E, S, z, r, T, vaccines, f_interval, learning_rate_change

            while True:
                if T % vaccine_interval == 0:

                        vaccines_per_zone = resource_allocation(0.3, vaccines, T)
                        arr_infected, arr_suspected, arr_exposed = np.zeros(z), np.zeros(z), np.zeros(z)

                        for delta in range(z):
                            arr_infected[delta] = len([i for i in range(eG) if entities[i].state == 'I' and entities[i].zone == delta])
                            arr_suspected[delta] = len([i for i in range(eG) if entities[i].state == 'S' and entities[i].zone == delta])
                            arr_exposed[delta] = len([i for i in range(eG) if entities[i].state == 'E' and entities[i].zone == delta])

                            available_vaccine = vaccines_per_zone[delta]
                            arr = [iota for iota in range(len(agent_zones)) if ((agent_zones[iota]==delta and entities[iota].state != 'D') and (entities[iota].state != 'R' and entities[iota].state != 'I'))]

                            immune, vaccinated = [], []
                            for phi in range(len(arr)):
                                while(available_vaccine > 0):
                                    initial_state = entities[arr[phi]].state
                                    entities[arr[phi]].state = np.random.choice([initial_state, 'R'], size=1, p=[1-expectedR[delta], expectedR[delta]])[0]
                                    vaccinated.append(arr[phi])
                                    if(entities[arr[phi]].state == "R"):
                                        immune.append(arr[phi])
                                        if initial_state == 'S':
                                            arr_suspected[delta] -= 1
                                        elif initial_state == 'E':
                                            arr_exposed[delta] -= 1
                                    available_vaccine -= 1
                                    break

                            r[delta] = r[delta] + (((len(immune)/(len(vaccinated)+0.000000001)) - r[delta]) * learning_rate)

                        learning_rate_change.append(r[2]) 

                        I = np.array(arr_infected)
                        S = np.array(arr_suspected)
                        E = np.array(arr_exposed)

                yield self.env.timeout(minimumWaitingTime)

    def prospect(x, beta, lambdas):

        if x == 0:
            return 1

        return lambdas * math.pow(x, beta)


    def count_extra(A):

        how_many_used = 0
        extra = 0

        for i in range(A.shape[0]):
            print (A[i])

            if np.sum(A[i]) > 0:
                how_many_used += 1

                if np.sum(A[i]) < 1.0:
                    extra += 1.0 - np.sum(A[i])
        return how_many_used, extra

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

    def least_distance_per_cluster(C, arr):

        array = np.zeros((len(arr), len(arr)))
        for row in range(len(arr)):
            for column in range(len(arr)):
                array[row][column] = euclidean(C[arr[row]], C[arr[column]])
        avg_distance = (np.mean(array, axis=1)).reshape(len(arr), )
        least = np.amin(avg_distance)
        result = 0
        for et in range(len(avg_distance)):
            if avg_distance[et] == least:
                result = arr[et]
                break
        return result

    def def_zone_and_coor():

        global population_val, C

        population_val = np.array(file['Population'].values)
        pop = np.true_divide(population_val, np.sum(population_val))

        C = []
        coordinates = np.array(file['Location'].values)
        for a in range(0, len(coordinates)):
            arr = coordinates[a].split(', ')
            for b in range(0, len(arr)):
                arr[b] = float(arr[b])
            C.append((arr[1], arr[0]))
        agent_zones = np.random.choice(len(C), size=eG, p=pop)
        agent_initial_coordinates = [C[d] for d in agent_zones]

        return C, agent_zones, agent_initial_coordinates

    def radius():

        global population_val, file

        area = np.true_divide(np.array(file['Population'].values), np.array(file['Population Density'].values))
        rad = [math.sqrt(area[i]/math.pi) for i in range(len(area))] / ((np.max(population_val))/eG)
        return rad

    def initial_state():

        global infected_ratio, susceptible_ratio, exposed_ratio, population_val

        infected_ratio = np.true_divide(np.array(file['Total Infected'].values), np.array(file['Population'].values)) + infected_bias
        exposed_ratio = pe * (1 - infected_ratio)
        susceptible_ratio = 1 - (infected_ratio + exposed_ratio)
        initial_state = [(np.random.choice(['S', 'E', 'I'], size=1, 
                                   p=[susceptible_ratio[(agent_zones[c])],
                                      exposed_ratio[(agent_zones[c])],
                                      infected_ratio[(agent_zones[c])]])[0]) for c in range(len(agent_zones))]
        return initial_state

    def initial_state():

        global infected_ratio, susceptible_ratio, exposed_ratio, population_val

        infected_ratio = np.true_divide(np.array(file['Total Infected'].values), np.array(file['Population'].values)) + infected_bias
        exposed_ratio = pe * (1 - infected_ratio)
        susceptible_ratio = 1 - (infected_ratio + exposed_ratio)
        initial_state = [(np.random.choice(['S', 'E', 'I'], size=1, 
                                   p=[susceptible_ratio[(agent_zones[c])],
                                      exposed_ratio[(agent_zones[c])],
                                      infected_ratio[(agent_zones[c])]])[0]) for c in range(len(agent_zones))]
        return initial_state

    def resource_allocation(p, T, time):

        global trade_off, B, N, I, E, S, C, r

        # Low trade-off favor economic and high trade-off favors vaccine formulation
        trade_off = 0.95

        how_many = warehouse

        kmeans = KMeans(n_clusters=warehouse, random_state=0).fit(C)
        cluster_center = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        label_arr = [[] for alpha in range(warehouse)]
        for beta in range(len(cluster_labels)):
            label_arr[cluster_labels[beta]].append(beta)

        # List of warehouses
        LW = [least_distance_per_cluster(C, label_arr[gamma]) for gamma in range(warehouse)]
        print('LW: ', LW)

        # Equally distributing vaccines across warehouse zones
        VW = {}
        current_warehouse = 0
        for f in range(1, T + 1):
            VW[f - 1] = LW[current_warehouse]
            if (f % (T / warehouse) == 0):
                current_warehouse += 1

        # Defining the parameters for the optimization
        B = np.array(file['Population Density'].values) * p  # rate of disease spread
        N = np.array(num_agent_per_zone) # total population for each zone
        Z_B = [((B[i]-np.mean(B))/np.sum(B))+1.0 for i in range(z)]
        
        if time <= vaccine_interval:
            r = np.array([0.4 for app in range(z)])
            I = np.array([infected_ratio[t]*num_agent_per_zone[t] for t in range(len(N))])
            E = (N-I) * pe
            S = N - (I + E)

        ir = [I[i]/(N[i]+0.000001) for i in range(z)]
        Z_I = [((ir[i]-np.mean(ir))/np.sum(ir))+1.0 for i in range(z)]

        model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)
        X = pulp.LpVariable.dicts("X", ((i, j) for i in range(warehouse) for j in range(len(B))), lowBound = 0.0, upBound = int(T/warehouse), cat='Continuous')

        dist_array = [geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)]
        max_dist = np.max(dist_array)
        den_economic = float(T * max_dist)
        model += np.sum([X[j, b] * geodesic(C[LW[j]], C[b]).miles for j in range(warehouse) for b in range(z)])/den_economic

        
        # Constraint 1 --------------------------------------------------------------------------
        
        for i in range(warehouse):
            # Condition 1: If you must assign all the vaccines generated by a warehouse
            model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) == int(T/warehouse)
            
            # Condition 2: If you want to minimize the number of vaccines
            #model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) <= int(T/warehouse)

        # Constraint 2 --------------------------------------------------------------------------
    
        s = 0.0
        for i in range(warehouse):
            s += pulp.lpSum([X[(i, j)] for j in range(len(B))])
                
        # Condition 1: If you must assign all the vaccines generated by a warehouse
        model += s == T

        # Condition 2: If you want to minimize the number of vaccines
        #model += s <= T
                
        # Constraint 3 (fairness lower) ---------------------------------------------------------
        
        for i in range(len(B)):
            # Condition 3: If calculation is based on susceptible population
            c = (S[i] - r[i] * pulp.lpSum([X[(j, i)] for j in range(warehouse)]))/sum(S)
            
            # Condition 4: To include population density
            c *= Z_B[i]
            
            # Condition 5: To include infected population
            c *= Z_I[i]
            
            model += pulp.lpSum([X[(j, i)] for j in range(warehouse)]) >= trade_off * c * T
        
        # Condition 3 only
        #another_arr = [(S[i])/(max(S)) for i in range(z)]
        
        # Condition 3 + 4
        #another_arr = [(S[i]*B[i])/(max(S)*max(B)) for i in range(z)]
    
        # Condition 3 + 4 + 5
        another_arr = [(S[i]*ir[i]*B[i])/(np.max(S)*np.max(ir)*np.max(B)) for i in range(z)]
        
        # -----------------------------------------------------------------------------------------
        
        model.solve()
        print(pulp.LpStatus[model.status])
        
        
        # Transferred the pulp decision to the numpy array (A)
        A = np.zeros((T, len(B)))
        for i in range(warehouse):
            for j in range(len(B)):
                A[i, j] = X[(i, j)].varValue
        
        global double_arr
        double_arr.append(["Value: " + str(pulp.value(model.objective))])
        
        vaccines_per_zone = []
        for i in range(len(B)):
            vaccines_per_zone.append(np.sum(A[:, i]))
        
        print('Trade-off value: ' + str(trade_off))
        plt.figure(figsize=(10,4))
        plt.bar([i for i in range(z)], vaccines_per_zone)
        plt.show()
        
        # ----------------------------------------------------------------------------------------
        
        '''
        print('\nformula', another_arr)
        print('\nvac', vaccines_per_zone)
        slope, intercept, r_value, p_value, std_err = stats.linregress(vaccines_per_zone,another_arr)
        print('Correlation: ', np.corrcoef(vaccines_per_zone, another_arr)[0, 1])
        print('R-Value: ', r_value)

        plt.figure(figsize=(8,4))
        plt.plot([0, np.max(vaccines_per_zone)], [intercept, (slope*np.max(vaccines_per_zone) + intercept)], col
        plt.scatter(vaccines_per_zone, another_arr, s=5)
        plt.ylabel('Susceptible, Population Density, and Infected Score', fontsize=8)
        plt.xlabel('Vaccines Per Zone', fontsize=14)
        plt.tight_layout()
        plt.savefig('susceptible_pop_density_and_infected.png', dpi=300)
        plt.show()
        '''
        
        return np.array(vaccines_per_zone)


    ''' Variables and Parameters for Simulation ''' 

    # Create Simpy environment and assign nodes to it. 
    env = simpy.Environment()

    # Susceptible-> Exposed
    alpha = 0.2

    # Susceptible-> Infected
    beta = 0.2

    # Exposed-> Infected
    gamma = 0.2

    # Infected-> Recovered
    delta = 0.2

    # Infected-> Death
    sigma = 0.2

    PT = 10

    # Fraction of susceptible/exposed nodes
    pe = 0.3

    # Simulation area --> not relevant right now
    Xlim, Ylim = 100, 100

    # Number of agents, zones, warehouses, and vaccines (optimization parameters)
    eG = 200
    z = 45
    warehouse = 5
    vaccines = 100

    # Simulation time-variable
    T = 0

    # Simulation duration
    Duration = 5

    # Move how often
    mho = 3

    # Time intervals for administering vaccines
    vaccine_interval = 5

    # Minimum waiting time
    minimumWaitingTime = 1

    # Variable used to increase proportion of infected (in case it's too low)
    infected_bias = 0.0 

    sensing_range = 30

    # File used for importing data
    file = pd.read_csv("covid_confirmed_NY_july.csv")
    file = file.iloc[0:z,:]

    # Initial position and coordinates of node based on population density likelihood PW 
    zone_coordinates, agent_zones, Coor = def_zone_and_coor()

    # Scaled radius of each zone
    R = radius()

    # Number of agents in each zone
    num_agent_per_zone = [0 for i in range(z)]
    for epsilon in range(len(agent_zones)):
        num_agent_per_zone[agent_zones[epsilon]] += 1

    #print("Number of agents per zone: ", num_agent_per_zone)    

    # Learning rate variables
    expectedR = [0.2 for var in range(z)]
    learning_rate = 0.15
    learning_rate_change.append(0.4)

    # List of node initial states
    STATE = initial_state()

    entities = [Node(env, i, Coor[i], STATE[i], alpha, beta, gamma, delta, sigma, agent_zones[i]) for i in range(eG)]
    env.run(until = Duration)
   

