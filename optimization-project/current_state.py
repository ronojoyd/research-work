#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import random
import networkx as nx
import simpy
import numpy as np
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
double_arr = []
learning_rate_change = []
    
for i in range(1):
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
            C.append([arr[1], arr[0]])
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
        trade_off = 0.1

        how_many = warehouse

        kmeans = KMeans(n_clusters=warehouse, random_state=0).fit(C)
        cluster_center = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        label_arr = [[] for alpha in range(warehouse)]
        for beta in range(len(cluster_labels)):
            label_arr[cluster_labels[beta]].append(beta)

        # List of warehouses
        LW = [least_distance_per_cluster(C, label_arr[gamma]) for gamma in range(warehouse)]
        LW = np.array(LW)
        print('\nLW:', LW, '\n')

        # Equally distributing vaccines across warehouse zones
        VW = {}
        current_warehouse = 0
        for f in range(1, T + 1):
            VW[f - 1] = LW[current_warehouse]
            if (f % (T / warehouse) == 0):
                current_warehouse += 1
        #print(VW.values())
        
        # Defining the parameters for the optimization
        B = np.array(file['Population Density'].values) * p  # rate of disease spread
        N = np.array(num_agent_per_zone) # total population for each zone
        if time <= vaccine_interval:
            r = np.array([0.4 for app in range(z)])
            I = np.array([infected_ratio[t]*num_agent_per_zone[t] for t in range(len(N))])  
            E = (N-I) * pe
            S = N - (I + E) 

        ir = I/(N+0.00001)
        
        dist_array = [geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)]
        max_dist = np.max(dist_array)
        den_economic = float(T * max_dist)
        den_infected = float(np.max(S) * np.max(B)*z)
        
        
        
        # CP-SAT solver
        '''
        model = cp_model.CpModel()
        var_upper_bound = (math.floor(T/warehouse))
        X = [[model.NewIntVar(0, var_upper_bound, 'X_w' + str(j) + '_z' + str(b)) for b in range(z)] for j in range(warehouse)]
        
        model.Add(np.sum(X) == T)
        for j in range(warehouse):
            model.Add(sum([X[j][b] for b in range(z)]) == int(T/warehouse))
        
        model.Minimize(sum([X[j][b] * geodesic(C[VW[j]], C[b]).miles for j in range(warehouse) for b in range(z)]))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        print(status)
        '''

        # MIP Solver
        '''
        solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        infinity = solver.infinity()

        X = [[solver.IntVar(10.0, infinity, 'X_' + str(j) + '_' + str(b)) for b in range(z)] for j in range(warehouse)]
                
        solver.Add(np.sum(X) == T)
        for i in range(warehouse):
            solver.Add(sum([X[i][j] for j in range(z)]) == int(T/warehouse))
        
        solver.Maximize(sum([X[j][b] * geodesic(C[VW[j]], C[b]).miles for j in range(warehouse) for b in range(z)])/den_economic)
        
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print('Solution:')
            print('Objective value =', solver.Objective().Value())
        else:
            print('The problem does not have an optimal solution.')
        '''   
        
        
        '''
        model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)
        X = pulp.LpVariable.dicts("X", ((i, j) for i in range(warehouse) for j in range(len(B))), lowBound = 0.0, upBound = int(T/warehouse), cat='Continuous')
        
        model += np.sum([X[j, b] * geodesic(C[VW[j]], C[b]).miles for j in range(warehouse) for b in range(z)])/den_economic
        
        # Constraint 1 --------------------------------------------------------------------------
        for i in range(warehouse):
            # Condition 1: If you must assign all the vaccines generated by a warehouse
            #model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) == int(T/warehouse)

            # Condition 2: If you want to minimize the number of vaccines
            model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) <= int(T/warehouse)

        # Constraint 2 --------------------------------------------------------------------------
        s = 0.0
        for i in range(warehouse):
            s += pulp.lpSum([X[(i, j)] for j in range(len(B))])

        # Condition 1: If you must assign all the vaccines generated by a warehouse
        #model += s == T

        # Condition 2: If you want to minimize the number of vaccines 
        model += s <= T

        # Constraint 3 (fairness lower) ---------------------------------------------------------
        for i in range(len(B)):

            # Condition 3: If calculation is based on susceptible population
            c = (S[i] - r[i] * pulp.lpSum([X[(j, i)] for j in range(warehouse)]))/sum(S)

            # Condition 4: To include population density
            #c = c * B[i]/np.median(B)

            # Condition 5: To include infected population
            #c = c * ir[i]/np.median(ir)

            model += pulp.lpSum([X[(j, i)] for j in range(warehouse)]) >= trade_off * c * T
        

        # Transferred the pulp decision to the numpy array (A)
        A = np.zeros((T, len(B)))
        for i in range(warehouse):
            for j in range(len(B)):
                A[i, j] = X[(i, j)].varValue
        '''
        

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
    eG = 2000
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

    print("Number of agents per zone: ", num_agent_per_zone)    

    # Learning rate variables
    expectedR = [0.2 for var in range(z)]
    learning_rate = 0.1
    learning_rate_change.append(0.4)

    # List of node initial states
    STATE = initial_state()

    resource_allocation(0.3, vaccines, T)
    
    print('\nWe live in a twilight world...')


# In[ ]:




