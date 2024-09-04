#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install simpy')
get_ipython().system('pip install geopy')
get_ipython().system('pip install ortools')


# In[24]:


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
        
        # Defining the parameters for the optimization
        B = np.array(file['Population Density'].values) * p  # rate of disease spread
        N = np.array(num_agent_per_zone) # total population for each zone
        if time <= vaccine_interval:
            r = np.array([0.4 for app in range(z)])
            I = np.array([infected_ratio[t]*num_agent_per_zone[t] for t in range(len(N))])  
            E = (N-I) * pe
            S = N - (I + E) 
        
        den_infected = float(np.max(S) * np.max(B)*z)
        dist_array = [geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)]
        max_dist = np.max(dist_array)
        den_economic = float(T * max_dist)
        
        solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        
        # Defining the decision variable
        x = {}
        for i in range(T):
            for j in range(z):
                x[i, j] = solver.IntVar(0, 1, '')
        
        # Constraints
        solver.Add(solver.Sum([x[i, j] for i in range(T) for j in range(z)]) == T)
        for i in range(T):
            solver.Add(solver.Sum([x[i, j] for j in range(z)]) == 1)
        for j in range(z):
            solver.Add(solver.Sum([x[i, j] for i in range(T)]) >= 1)
        
        objective_terms = []
        for i in range(T):
            for j in range(z):
                # objective_terms.append(B[j] * ((I[j] + 0.001) / (N[j] + 0.001)) * (S[j] - (r[j] * x[i, j]))) # Vaccination
                
                objective_terms.append(geodesic(C[VW[i]], C[j]).miles * x[i, j]) # Economic
        
        #solver.Minimize(solver.Sum(objective_terms)/den_infected) # Vaccination
        
        solver.Minimize(solver.Sum(objective_terms)/den_economic) # Economic
        
        status = solver.Solve()
        print('Objective function value: ', solver.Objective().Value())
        
        A = [[0 for i in range(z)] for i in range(T)]
        for i in range(T):
            for j in range(z): 
                A[i][j] = x[i, j].solution_value()
        
        vaccines_per_zone = np.sum(A, axis=0)
        plt.figure(figsize=(10, 4))
        plt.bar([i for i in range(z)], vaccines_per_zone)
        plt.show()
        
        return vaccines_per_zone
        

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




