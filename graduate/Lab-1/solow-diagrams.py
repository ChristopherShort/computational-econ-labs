import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# grid of points to plot
grid = np.linspace(0, 10, 1000)

def f(k):
    """Intensive form of a Cobb-Douglas production function."""
    return k**alpha

def u(C):
    """CRRA utilty function."""
    if theta != 1.0:
    	return (C**(1-theta) - 1) / (1 - theta)
    else: 
        return np.log(C)
            
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

for alpha in [0.25, 0.5, 0.75]:
    ax.plot(grid, f(grid), label=r'$\alpha=%g$' %alpha)
 
ax.set_xlabel('Capital per effective worker, k', fontsize=15, family='serif') 
ax.set_xticks([])
ax.set_ylabel('f(k)', fontsize=15, family='serif', rotation='horizontal') 
ax.yaxis.set_label_coords(-0.05, 0.5)
ax.set_yticks([])
ax.set_title('Intensive form of a Cobb-Douglas production technology', 
             fontsize=15, family='serif')
ax.legend(loc=0, frameon=False)
plt.savefig('graphics/production-function-intensive-form.png')
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

colors = mpl.cm.jet(np.linspace(0, 1, 10))

for i, theta in enumerate(np.logspace(-2, 1, 10)):
    ax.plot(grid, u(grid), color=colors[i], label=r'$\theta=%.2f$' %theta)
 
ax.set_xlabel('Consumption per worker, C', fontsize=15, family='serif') 
ax.set_xticks([])
ax.set_ylabel('u(C)', fontsize=15, family='serif', rotation='horizontal') 
ax.yaxis.set_label_coords(-0.05, 0.5)
ax.set_yticks([])
ax.set_ylim(-10, 10)
ax.set_title('Households have CRRA utility', fontsize=15, family='serif')
ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.025, 1))
plt.savefig('graphics/CRRA-preferences.png', bbox_inches='tight')
plt.show()