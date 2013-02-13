"""Toy dynamic programming problem solved using Value iteration.  
Although the implementation is in pure Python it is blindingly fast!

"""
import matplotlib.pyplot as plt

def d_infty(v, w):
    """d_infty metric"""
    return max(abs(v[x] - w[x]) for x in S)

def U(c):
    """CRRA Utility function"""
    return (c**(1 - theta) - 1) / (1 - theta)

def q(z):
    """Probability mass function, uniform distribution"""
    return 1.0 / len(Z) if 0 <= z <= B else 0

def Gamma(x):
    """The correspondence of feasible actions"""
    return range(min(x, M) + 1)

def T(v):
    """An implementation of the Bellman Operator!
    
    Parameters:
        
        v, a sequence representing a function defined on S
    
    Returns:
        
        Tv, a list

    """
    Tv = []
    for x in S:
        # Compute the value of the objective function for each
        # a in Gamma(x), and store result in vals
        vals = []
        for a in Gamma(x):
            y = U(x - a) + beta * sum(v[a + z] * q(z) for z in Z)
            vals.append(y)
            # Store the maximum reward for this x in the list Tv
        Tv.append(max(vals))
    return Tv

def greedy(v):
    """Computes the optimal (i.e., greedy policy) for the 
    value function v.

    """
    g = []
    for x in S:
        runningmax = -1
        for a in Gamma(x):
            y = U(x - a) + beta * sum(v[a + z] * q(z) for z in Z)
            if y > runningmax:
                runningmax = y
                maximizer = a
        g.append(maximizer)
    return g

##### Value iteration scheme #####

# define utility parameters...
theta = 0.5
beta = 0.95 

# upper bound on size of fish catch
B = 10 

# storage capacity
M = 5

# define the state and shock space
S = range(B + M + 1)
Z = range(B + 1)        

# initial guess for value function
old_V = [U(x) for x in S]

# specify some tolerance...
tol = 0.0001

#... and initialize dist
dist   = 1.0
n_iter = 0

# Value iteration algorithm
while dist > tol:
    new_V = T(old_V)
    dist  = d_infty(new_V, old_V)
    if n_iter % 10 == 0:
        print "After", n_iter, "iterations, the distance is",  dist
    old_V = new_V
    n_iter += 1
    
print "Convergence after", n_iter, "iterations!"

# compute the optimal policy
pol = greedy(new_V)

##### Plot the resulting value and policy functions #####
fig = plt.figure(figsize=(12,6))

# plot the value function
ax1 = fig.add_subplot(121)
ax1.plot(S, new_V, marker='o', linestyle='--')

# axes, labels, title, etc
ax1.set_xlabel("Stock of fish, $X_t$", fontsize=15)
ax1.set_ylabel("$V^*(X_{t})$", rotation="horizontal", fontsize=15)
ax1.set_ylim(0, 55)
ax1.set_title("Value function for Colonel Kurtz", fontsize=15)

# plot the optimal policy
ax2 = fig.add_subplot(122)
ax2.plot(S, pol, marker='o', linestyle='--', color='g')

# axes, labels, title, etc
ax2.set_xlabel("Stock of fish, $X_t$", fontsize=15)
ax2.set_ylabel("$\sigma^*(X_{t})$", rotation="horizontal", fontsize=15)
ax2.set_ylim(0, M + 1)
ax2.set_title("Policy function for Colonel Kurtz", fontsize=15)

# tighten up the layout and display
fig.tight_layout()
plt.savefig('Graphics/Kurtz-solution.png')

##### A cool plot of policy function iterates #####

current = [U(x) for x in S]
tolerance = 0.001

plt.figure(figsize=(8,6))

# value iteration
while 1:
    plt.plot(greedy(current), 'go--', alpha = 0.15)
    new = T(current)
    if d_infty(new, current) < tolerance:
        break
    current = new

# compute the optimal policy
g = greedy(new)

# axes, labels, etc
plt.xlabel("Stock of fish, $X_t$", fontsize=15)
plt.ylabel("$\sigma^*(X_{t})$", rotation="horizontal", fontsize=15)
plt.ylim(0, M + 1)
plt.title("Policy iterates for Colonel Kurtz", fontsize=15)

print 'The optimal policy is:'
print g

# display
plt.show()
    

