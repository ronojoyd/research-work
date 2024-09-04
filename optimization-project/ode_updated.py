import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# function that returns dz/dt
def model(z, t):

    global p, alpha
    dsdt = - beta * z[0] * z[2]
    dedt = beta * z[0] * z[2] - sigma * z[1]
    didt = sigma * z[1] - gamma * z[2]
    drdt = gamma * alpha * z[2]
    dddt = sigma * (1 - alpha) * z[2]

    print (bool(dsdt + dedt + didt + drdt + dddt < 0.01))
    return [dsdt, dedt, didt, drdt, dddt]


# Contact rate
beta = 0.00001

# Exposed-> Infected
sigma = 0.01

# Infected-> Death/Recovered
gamma = 0.3

# Infected to recovered
alpha = 0.9

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