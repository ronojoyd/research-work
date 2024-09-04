import matplotlib.pyplot as plt
import pickle

L0 = pickle.load(open('En-0.p', 'rb'))
L1 = pickle.load(open('En-1.p', 'rb'))

plt.plot([i for i in range(len(L0))], L0, color = 'red', label = 'Low power')
plt.plot([i for i in range(len(L1))], L1, color = 'green', label = 'High power')

plt.xlabel('Time in minutes', fontsize = 15)
plt.ylabel('Mean Residual Energy in Joules', fontsize = 15)
plt.legend(prop={'size': 15})
plt.savefig('En-Res.png', dpi = 300)
plt.show()


