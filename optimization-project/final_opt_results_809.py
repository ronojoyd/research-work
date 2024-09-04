#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from sklearn.metrics import silhouette_score, calinski_harabasz_score

final_arr = []
learning_rate_change = []
double_arr = []

for i in range(1): 

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

        '''
        #Add the function to generate warehouses
        array = np.zeros((z, len(LW)))
        for row in range(z):
            for column in range(len(LW)):
                array[row][column] = euclidean(C[row], C[LW[column]])
        '''

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

        '''    
        B = np.array([i + 1 for i in range(z)])
        N = np.array([z - i for i in range(z)])
        I = np.array([random.uniform(1, N[i]) for i in range(z)])
        E = (N-I) * pe
        S = N - (I + E) 


        B = np.array([2 for i in range(z)])
        N = np.array([100 for i in range(z)])
        I = np.array([25 for i in range(z)])
        E = (N-I) * pe
        S = N - (I + E)
        '''

        ir = I/(N+0.00001)

        model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)
        X = pulp.LpVariable.dicts("X", ((i, j) for i in range(warehouse) for j in range(len(B))), lowBound = 0.0, upBound = int(T/warehouse), cat='Continuous')

        dist_array = [geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)]
        max_dist = np.max(dist_array)
        den_economic = float(T * max_dist)
        model += np.sum([X[LW.index(VW[j]), b] * geodesic(C[VW[j]], C[b]).miles for j in range(T) for b in range(z)])/den_economic

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

        # ---------------------------------------------------------------------------------------

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
        
        global final_arr
        final_arr.append(np.sum(vaccines_per_zone))
        
        plt.figure(figsize=(10,4))
        plt.bar([i for i in range(z)], vaccines_per_zone)
        print(vaccines_per_zone)
        #plt.tight_layout()
        #plt.savefig('plot.png', dpi=300)
        plt.show()
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
    Duration = 50

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
    learning_rate = 0.15
    learning_rate_change.append(0.4)

    # List of node initial states
    STATE = initial_state()

    entities = [Node(env, i, Coor[i], STATE[i], alpha, beta, gamma, delta, sigma, agent_zones[i]) for i in range(eG)]
    env.run(until = Duration)
    
    print(final_arr)
    


# In[11]:


# Learning Rate Results

lr0_1 = [0.4, 0.40999999999375003, 0.3832857142780485, 0.35924285713391707, 0.36617571427154577, 0.34384385712806464, 0.3094594714152582, 0.29279923855740586, 0.2920907432690122, 0.32002452607680487, 0.3023077877527979, 0.30064843754486503, 0.284869308074052, 0.2992395201176672, 0.28360128238957394, 0.25524115415061654, 0.25828846730290184, 0.24674533485628514, 0.2595708013659691, 0.24789943551304566, 0.23739520624541458, 0.2511556856161856, 0.22604011705456706, 0.22843610534598535, 0.2055924948113868, 0.19753324532868563, 0.19027992079425457, 0.20875192871014162, 0.20037673583756496, 0.18033906225380847, 0.18730515602530262, 0.19357464041964736, 0.21171717637299514, 0.21554545873257064, 0.21899091285618857, 0.22209182156744472, 0.21238263940913776, 0.20364437546666148, 0.20827993791687033, 0.2249519441204958, 0.2024567497084462]
lr0_2 = [0.4, 0.32, 0.330999999990625, 0.31479999998625, 0.30183999998275, 0.316471999976825, 0.30317759997521, 0.292542079973918, 0.2590336639760094, 0.23222693117768253, 0.260781544932771, 0.2836252359368418, 0.2769001887432234, 0.2715201509883287, 0.21721612079066296, 0.22377289662628036, 0.2540183172916493, 0.30321465382081947, 0.26757172305353055, 0.2890573784334494, 0.28124590274050953, 0.22499672219240763, 0.2549973777445511, 0.2289979021925159, 0.23319832174776273, 0.26155865738883516, 0.2592469259048181, 0.2323975407207295, 0.2109180325734586, 0.21873442605251686, 0.19998754083888848, 0.18221225489086387, 0.21243647057195036, 0.19217139867731337, 0.17595934116160378, 0.1629896951490361, 0.17483620055873506, 0.18431340488649423, 0.16967294612894845, 0.18018280134266493, 0.16636846329388502]
lr0_002 = [0.4, 0.40019999999987504, 0.3993995999998753, 0.3996008007997505, 0.3990515991981198, 0.3985034959996923, 0.39820648900763045, 0.3981600760295214, 0.39761375587743114, 0.39731852836561377, 0.39677389130885127, 0.3962303435262023, 0.3954378828391499, 0.3950914515178666, 0.39474571305922596, 0.39417844385530504, 0.394056753634187, 0.39393530679351124, 0.393397436179893, 0.39305508575192827, 0.3924911978026219, 0.39215065985141173, 0.391810802976104, 0.39124940359234933, 0.3906891270073622, 0.3899077487533475, 0.38957237770023584, 0.38923767738923043, 0.3886814242566495, 0.3883485058525312, 0.3880162532852212, 0.3874624430008483, 0.38668751811484664, 0.38613636530081447, 0.38596409257015285, 0.3851921643850125, 0.3846217800562225, 0.38465253649603004, 0.384283231422998, 0.38351466496015196, 0.38294763563021167]

plt.plot([i for i in range(len(lr0_1))], lr0_1, color='red', label='lr = 0.1')
plt.plot([i for i in range(len(lr0_2))], lr0_2, color='blue', label='lr = 0.2')
plt.plot([i for i in range(len(lr0_002))], lr0_002 , color='green', label='lr = 0.002')
plt.plot([i for i in range(len(lr0_002))], [0.2 for i in range(len(lr0_002))], linestyle='--', color='black', label='expected lr')
plt.ylabel('Learning Rate', fontsize=14)
plt.xlabel('Duration', fontsize=14)
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig('learning_rate_630.png', dpi=300)
plt.show()


# In[10]:


# Experiment 1

p0_1 = [0.14747104, 0.049438067, 14.685921, 0.049310038, 0.0, 0.049428155, 0.09890848, 0.098838255, 0.049349479, 0.14833083, 0.049112285, 0.049453712, 0.049403661, 0.4879264, 0.83476175, 0.0, 19.851669, 0.0, 18.130565, 0.0, 0.0, 0.0, 0.098927632, 1.5474513, 0.0, 0.0, 0.098504834, 0.59092776, 19.416843, 0.52771238, 0.9250142, 0.24601829, 0.14739685, 19.210535, 0.049380366, 0.33684811, 0.0, 0.14825599, 0.049434171, 0.04883354, 1.3461045, 0.0, 0.19218593, 0.18973888, 0.0]
p0_95 = [1.1307041, 0.0, 8.0685203, 0.37807464, 0.37894179, 0.75796056, 1.1375408, 0.37891106, 0.0, 0.37909878, 0.37655841, 0.0, 0.0, 1.4964304, 4.1414127, 0.0, 3.3494888, 0.0, 8.6787166, 0.0, 0.0, 0.0, 0.3792537, 11.864766799999998, 0.0, 0.75755743, 0.0, 3.0205452, 0.0, 5.5174457, 7.0923574, 0.75451799, 1.1301352, 16.596757, 0.0, 2.582714, 1.1307673, 0.3789075, 1.1370792, 0.74884238, 10.320981, 1.8911193, 2.2103195, 1.4547841, 0.37878946]

plt.figure(figsize=(15,5))
plt.plot([i for i in range(45)], p0_1, color='red', label='trade-off = 0.10', linestyle='--', marker='o')
plt.plot([i for i in range(45)], p0_95, color='blue', label='trade-off = 0.95', linestyle='--', marker='o')
plt.ylabel('Number of Vaccines', fontsize=14)
plt.xlabel('Zone ID', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('experiment_condition_1_3.png', dpi=300)
plt.show()


# In[12]:


ra = [74.4014732, 74.03422379999999, 73.16738593000001, 70.413396355, 68.99474115000001, 63.222150141, 58.8071122, 56.967853000000005, 53.5003867, 50.947497600000005]

plt.plot([i*5 for i in range(len(ra))], ra, color='red', label='trade-off = 0.95', linestyle='--', marker='o')
plt.ylabel('Sum of Vaccines', fontsize=14)
plt.xlabel('Duration', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('experiment_condition_2_3.png', dpi=300)
plt.show()


# In[ ]:




