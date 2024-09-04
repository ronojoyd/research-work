#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pulp
import random
import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import operator
import itertools
import pandas as pd
import decimal
from geopy import distance
from scipy import optimize
from sklearn.cluster import KMeans
from scipy.integrate import odeint
from scipy.spatial.distance import *
from scipy import stats
import matplotlib.pyplot as plt
from geopy.distance import geodesic

for iteration in range(1):
    
    # Variable that tracks time in this program
    global_time = 0
    
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

    def beta_values():
        global file, mid_B
        B = np.array(file['Population Density'].values)
        beta_arr = [((B[i] - np.mean(B)) / np.sum(B) + 1.0) * mid_B for i in range(z)]
        return beta_arr
    
    
    def LW_func():
        
        global file, C
        
        C = []
        coordinates = np.array(file['Location'].values)
        for a in range(0, len(coordinates)):
            arr = coordinates[a].split(', ')
            for b in range(0, len(arr)):
                arr[b] = float(arr[b])
            C.append((arr[1], arr[0]))
            
        kmeans = KMeans(n_clusters=warehouse, random_state=0).fit(C)
        cluster_center = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        label_arr = [[] for alpha in range(warehouse)]
        for i in range(len(cluster_labels)):
            label_arr[cluster_labels[i]].append(i)

        # List of warehouses
        LW = [least_distance_per_cluster(C, label_arr[i]) for i in range(warehouse)]
        
        return LW
            
    
    def manual_ode_with_optimization(T):
        
        global global_time, alpha, beta, sigma, gamma, trade_off, infected_arr
        
        if global_time == 0:
            
            global S, E, I, R, D, N
            
            # N describes the population in each zone
            N = np.array(file['Population'].values)
            
            # Initial infected, exposed, and susceptible values for each zone
            I = np.array(file['Total Infected'].values)
            E = pe * (N - I)
            S = N - (I + E)
            R, D = [0 for i in range(z)], [0 for i in range(z)]
        
        else:
            
            for i in range(z):
                S[i] += (-beta[i] * S[i] * I[i])/N[i]
                E[i] += (beta[i] * S[i] * I[i])/N[i] - (sigma * E[i])
                I[i] += sigma * E[i] - gamma * I[i]
                R[i] += gamma * (1-alpha) * I[i]
                D[i] += sigma * alpha * I[i]
        
        for i in range(z):
            infected_arr[i].append(I[i])

        # List of warehouses
        LW = LW_func()

        # Equally distributing vaccines across warehouse zones
        VW = {}
        current_warehouse = 0
        for f in range(1, T + 1):
            VW[f - 1] = LW[current_warehouse]
            if (f % (T / warehouse) == 0):
                current_warehouse += 1

        Z_B = [((beta[i]-np.mean(beta))/np.sum(beta))+1.0 for i in range(z)]
        ir = [I[i]/(N[i]+0.000001) for i in range(z)]
        Z_I = [((ir[i]-np.mean(ir))/np.sum(ir))+1.0 for i in range(z)]
        
        model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)
        X = pulp.LpVariable.dicts("X", ((i, j) for i in range(warehouse) for j in range(z)), lowBound = 0.0, upBound = int(T/warehouse), cat='Continuous')

        dist_array = [geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)]
        max_dist = np.max(dist_array)
        den_economic = float(T * max_dist)
        model += np.sum([X[j, b] * geodesic(C[LW[j]], C[b]).miles for j in range(warehouse) for b in range(z)])/den_economic
        
        # Constraint 1 --------------------------------------------------------------------------
        
        for i in range(warehouse):
            # Condition 1: If you must assign all the vaccines generated by a warehouse
            #model += pulp.lpSum([X[(i, j)] for j in range(z)]) == int(T/warehouse)
            
            # Condition 2: If you want to minimize the number of vaccines
            model += pulp.lpSum([X[(i, j)] for j in range(z)]) <= int(T/warehouse)

        # Constraint 2 --------------------------------------------------------------------------
    
        s = 0.0
        for i in range(warehouse):
            s += pulp.lpSum([X[(i, j)] for j in range(z)])
                
        # Condition 1: If you must assign all the vaccines generated by a warehouse
        #model += s == T

        # Condition 2: If you want to minimize the number of vaccines
        model += s <= T
                
        # Constraint 3 (fairness lower) ---------------------------------------------------------
      
        for i in range(z):
            # Condition 3: If calculation is based on susceptible population
            # c = (S[i] - r[i] * pulp.lpSum([X[(j, i)] for j in range(warehouse)]))/sum(S)
            c = 1/z
            
            # Condition 4: To include population density
            # c *= Z_B[i]
            
            # Condition 5: To include infected population
            c *= Z_I[i]
            
            model += pulp.lpSum([X[(j, i)] for j in range(warehouse)]) >= trade_off * c * T
        
        # -----------------------------------------------------------------------------------------
        
        model.solve()
        print(pulp.LpStatus[model.status])
        
        # Transferred the pulp decision to the numpy array (A)
        A = np.zeros((T, z))
        for i in range(warehouse):
            for j in range(z):
                A[i, j] = X[(i, j)].varValue
                
        vaccines_per_zone = []
        for i in range(z):
            vaccines_per_zone.append(np.sum(A[:, i]))
        
        # Outputs the number of vaccines distributed to each zone
        print('LW: ', LW, '\nTrade-off value: ', str(trade_off))
        plt.figure(figsize=(10,4))
        plt.bar([i for i in range(z)], vaccines_per_zone)
        plt.show()
        
        global_time += 1
        
    
    # Number of agents, zones, warehouses, and vaccines (optimization parameters)
    eG = 200
    z = 45
    warehouse = 5
    vaccines = 100    
    
    trade_off = 0.95
    
    pe = 0.3
    
    Duration = 50
    
    infected_arr = [[] for i in range(z)]
        
    # File used for importing data
    file = pd.read_csv("covid_confirmed_NY_july.csv")
    file = file.iloc[0:z,:]
    
    # Median B value
    mid_B = 3.0
    
    # Parameters for transitioning between states
    beta = beta_values()
    sigma = 0.05
    gamma = 0.1
    alpha = 0.05
    
    while(global_time < Duration):
        manual_ode_with_optimization(vaccines)
        
    # This code displays how infected changes for each zone over time
    for i in range(z):
        plt.plot([i for i in range(Duration)], infected_arr[i])
    plt.show()


# In[ ]:




