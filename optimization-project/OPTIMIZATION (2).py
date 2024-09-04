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
import operator
import itertools
import pandas as pd
import decimal
from geopy import distance
from scipy import optimize
from sklearn.cluster import KMeans
from scipy.spatial.distance import *
from scipy import stats
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score

learning_rate_change = []
double_arr = []

for iteration in range(1): 

    class Node(object):

        def __init__(self, env, ID, coor, state, alpha, beta, gamma, sigma, zone):

            global T, d

            self.ID = ID
            self.env = env
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
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

            global minimumWaitingTime, beta

            while True:
                if T % PT == (self.ti + 1) % PT:

                    state_change = False

                    if self.state == 'S':
                        for u in self.nlist:
                            if entities[u].state == 'E' and random.uniform(0, 1) <= self.beta:
                                self.state = 'E'
                                state_change = True
                                break

                    if self.state == 'E' and state_change == False:
                        if random.uniform(0, 1) <= self.sigma:
                            self.state = 'I'
                            state_change = True

                    if self.state == 'I' and state_change == False:
                        if random.uniform(0, 1) <= self.gamma * (1-self.alpha):
                            self.state = 'R'
                            state_change = True

                    if self.state == 'I' and state_change == False:
                        if random.uniform(0, 1) <= self.gamma * self.alpha:
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

                        for zone in range(z):
                            arr_infected[zone] = len([i for i in range(eG) if entities[i].state == 'I' and entities[i].zone == zone])
                            arr_suspected[zone] = len([i for i in range(eG) if entities[i].state == 'S' and entities[i].zone == zone])
                            arr_exposed[zone] = len([i for i in range(eG) if entities[i].state == 'E' and entities[i].zone == zone])

                            available_vaccine = vaccines_per_zone[zone]
                            arr = [iota for iota in range(len(agent_zones)) if ((agent_zones[iota]==zone and entities[iota].state != 'D') and (entities[iota].state != 'R' and entities[iota].state != 'I'))]

                            immune, vaccinated = [], []
                            for phi in range(len(arr)):
                                while(available_vaccine > 0):
                                    initial_state = entities[arr[phi]].state
                                    entities[arr[phi]].state = np.random.choice([initial_state, 'R'], size=1, p=[1-expectedR[zone], expectedR[zone]])[0]
                                    vaccinated.append(arr[phi])
                                    if(entities[arr[phi]].state == "R"):
                                        immune.append(arr[phi])
                                        if initial_state == 'S':
                                            arr_suspected[zone] -= 1
                                        elif initial_state == 'E':
                                            arr_exposed[zone] -= 1
                                    available_vaccine -= 1
                                    break

                            r[zone] = r[zone] + (((len(immune)/(len(vaccinated)+0.000000001)) - r[zone]) * learning_rate)

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

    def resource_allocation(p, T, time):

        global trade_off, S, E, I, N, B, C, r, z, beta

        # Low trade-off favor economic and high trade-off favors vaccine formulation
        trade_off = 0.95

        how_many = warehouse

        kmeans = KMeans(n_clusters=warehouse, random_state=0).fit(C)
        cluster_center = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        label_arr = [[] for alpha in range(warehouse)]
        for i in range(len(cluster_labels)):
            label_arr[cluster_labels[i]].append(i)

        # List of warehouses
        LW = [least_distance_per_cluster(C, label_arr[i]) for i in range(warehouse)]

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
        Z_B = [((beta[i]-np.mean(beta))/np.sum(beta))+1.0 for i in range(z)]
        
        if time <= vaccine_interval:
            r = np.array([0.4 for i in range(z)])
            I = np.array([infected_ratio[t]*num_agent_per_zone[t] for t in range(len(N))])
            E = (N-I) * pe
            S = N - (I + E)

        ir = [I[i]/(N[i]+0.000001) for i in range(z)]
        Z_I = [((ir[i]-np.mean(ir))/np.sum(ir))+1.0 for i in range(z)]
        
        # Differential equations -----------------------------------

        # ----------------------------------------------------------

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
            #c *= Z_B[i]
            
            # Condition 5: To include infected population
            #c *= Z_I[i]
            
            model += pulp.lpSum([X[(j, i)] for j in range(warehouse)]) >= trade_off * c * T
        
        # Condition 3 only
        # another_arr = [(S[i])/(max(S)) for i in range(z)]
        
        # Condition 3 + 4
        # another_arr = [(S[i]*B[i])/(max(S)*max(B)) for i in range(z)]
    
        # Condition 3 + 4 + 5
        # another_arr = [(S[i]*ir[i]*B[i])/(np.max(S)*np.max(ir)*np.max(B)) for i in range(z)]
        
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
        
        # Outputs the number of vaccines distributed to each zone
        print('LW: ', LW)
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
    
    def beta_values():
        
        global file
        
        B = np.array(file['Population Density'].values)
        beta_arr = [3.0 + (((B[i]-np.amin(B))/(np.max(B)-np.amin(B))) * (5.7-3)) for i in range(z)]
        
        return beta_arr
        
        
    # Variables and Parameters for Simulation ------------------------------------ 

    # Create Simpy environment and assign nodes to it. 
    env = simpy.Environment()
    
    p = 0.1

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
    Duration = 200

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
    
    # Parameters for transitioning between states
    beta = beta_values()
    sigma = np.mean([1/3, 1/5])
    gamma = np.mean([1/5, 1/18])
    alpha = 0.05

    # Initial position and coordinates of node based on population density likelihood PW 
    zone_coordinates, agent_zones, Coor = def_zone_and_coor()

    # Scaled radius of each zone
    R = radius()

    # Number of agents in each zone
    num_agent_per_zone = [0 for i in range(z)]
    for epsilon in range(len(agent_zones)):
        num_agent_per_zone[agent_zones[epsilon]] += 1
    
    #print('Number of agents per zone: ', num_agent_per_zone)

    # Learning rate variables
    expectedR = [0.2 for var in range(z)]
    learning_rate = 0.002
    learning_rate_change.append(0.4)

    # List of node initial states
    STATE = initial_state()

    entities = [Node(env, i, Coor[i], STATE[i], alpha, beta[agent_zones[i]], gamma, sigma, agent_zones[i]) for i in range(eG)]
    env.run(until = Duration)
    
print(learning_rate_change)


# In[20]:


# LEARNING RATE PLOT

from sklearn.metrics import mean_absolute_error

p0_2 = [0.4, 0.3644444444395062, 0.33599999999111113, 0.33546666665214814, 0.2905955555414716, 0.32136533331218964, 0.2570922666497517, 0.27234047997906063, 0.2928723839738735, 0.2342979071790988, 0.2318827701827852, 0.22995066058573432, 0.2506271951278467, 0.22272397832203045, 0.2448458493168836, 0.24032112389301305, 0.23670134355391662, 0.2782499637221456, 0.3225999709652165, 0.3080799767659232, 0.27146398140961353, 0.2421711851245658, 0.19373694809965264, 0.19943400291922828, 0.1817694245551357, 0.14541553964410855, 0.22744354281405227, 0.2069548342481168, 0.18778608961824653, 0.21689553835385647, 0.21796087512259135, 0.1743687000980731, 0.18393940451796464, 0.19159596805387788, 0.1532767744431023, 0.12262141955448184, 0.13809713563958548, 0.17714437517092763, 0.18615994457624827, 0.19337240010050477, 0.1769201423001569]
p0_02 = [0.4, 0.39422222222197534, 0.39383777777659834, 0.3859610222210664, 0.3832418017760201, 0.38057696573987465, 0.38046542642413966, 0.37285611789565687, 0.3678989955374312, 0.3655410156260576, 0.360730195313224, 0.3585155914063345, 0.3538452795778953, 0.3542683739853999, 0.3521830065050669, 0.3501393463743406, 0.3481365594462288, 0.3436738282569917, 0.3418003516912269, 0.34246434465646486, 0.3381150577630231, 0.3338527566074501, 0.3321757014746761, 0.3280321874448701, 0.3239715436956602, 0.3197143350437223, 0.3199867150087738, 0.3135869807085983, 0.3139819077603523, 0.31436893627107115, 0.31558155754471223, 0.3167699263928805, 0.3129345278647104, 0.3091758373071037, 0.3054923205606491, 0.30604914081536205, 0.3049281579984298, 0.3013295948381487, 0.2978030029410732, 0.2918469428822517, 0.2904544484685573]
p0_002 = [0.4, 0.39942222222219753, 0.3990678222221482, 0.39849190879990143, 0.39791714720449917, 0.3971213129100902, 0.39654929250646753, 0.39597841614365215, 0.3958531259779574, 0.39550586417039657, 0.3947148524420558, 0.3939254227371717, 0.39353757189165733, 0.39315049674783403, 0.3927641957542984, 0.3923786673627498, 0.3917939100280043, 0.39181032220786827, 0.3914267015634125, 0.3906438481602857, 0.3902625604639251, 0.38988203534295723, 0.3899022712721913, 0.3895224667296069, 0.3889434217961277, 0.38916553495243544, 0.38858720388251056, 0.3882100294747055, 0.3876336094157361, 0.3874583421968446, 0.38668342551245094, 0.38671005866134606, 0.38633663854398337, 0.3859639652668554, 0.3851920373363217, 0.384967107807054, 0.3847971735913799, 0.3842275792441771, 0.38436821499469703, 0.38399947856466765, 0.3836314796074983]

plt.plot([i for i in range(len(p0_2))], p0_2, color='red', label='lr = 0.2')
plt.plot([i for i in range(len(p0_02))], p0_02, color='green', label='lr = 0.02')
plt.plot([i for i in range(len(p0_002))], p0_002, color='blue', label='lr = 0.002')
plt.plot([i for i in range(len(p0_002))], [0.2 for i in range(len(p0_002))], color='black', linestyle='--', label='expected lr')
plt.xlabel('Duration', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('lr.png', dpi=300)
plt.show()

print('Mean absolute error for lr of 0.2: ', mean_absolute_error([0.2 for i in range(len(p0_002))], p0_2))
print('Mean absolute error for lr of 0.02: ', mean_absolute_error([0.2 for i in range(len(p0_002))], p0_02))
print('Mean absolute error for lr of 0.002: ', mean_absolute_error([0.2 for i in range(len(p0_002))], p0_002))

