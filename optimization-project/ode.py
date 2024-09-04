import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# function that returns dz/dt
def model(z, t):

    global p, alpha
    dsdt = - alpha * z[0] * z[2]
    dedt = alpha * z[0] * z[2] - gamma * z[1]
    didt = gamma * z[1] - delta * z[2] - sigma * z[2]
    drdt = delta * z[2]
    dddt = sigma * z[2]

    print (bool(dsdt + dedt + didt + drdt + dddt < 0.01))
    return [dsdt, dedt, didt, drdt, dddt]


# Susceptible-> Exposed
alpha = 0.00001

# Susceptible-> Infected
beta = 0.01

# Exposed-> Infected
gamma = 0.01

# Infected-> Recovered
delta = 0.01

# Infected-> Death
sigma = 0.01

# initial condition
z0 = [1000000, 0, 10, 0, 0]

# time points
t = np.linspace(0, 50)

p = 0.1

# solve ODE
z = odeint(model, z0, t)
# print (z)

# plot results
plt.plot(t, z[:, 0], 'b-', label = 'Susceptible')
plt.plot(t, z[:, 1], 'r-', label = 'Exposed')
plt.plot(t, z[:, 2], 'g-', label = 'Infected')
plt.plot(t, z[:, 3], 'brown', label = 'Recovered')
plt.plot(t, z[:, 4], 'black', label = 'Death')

plt.show()