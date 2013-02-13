import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound, fsolve
from scipy.interpolate import interp1d

def u(state, c):
    """Agent has CRRA preferences where c is consumption per
    effective worker.

    In general, the flow of utility can depend on the state, but
    in this simple problem, the state variable does not enter 
    into the utility!
    
    """
    if theta != 1:
        return (c**(1 - theta)) / (1 - theta)
    else:
        return np.log(c)
    
def capital(k, c):
    """Equation of motion for capital per effective worker."""
    kplus = (1 / ((1 + g) * (1 + n))) * ((1 - delta) * k + k**alpha - c)
    return kplus

def k_star():
    """Steady state value of capital per effective worker."""
    kstar = (alpha * beta / ((1 + g)**theta - beta * (1 - delta)))**(1 / (1 - alpha))
    return kstar

def c_star():
    """Steady state value of consumption per effective worker."""
    cstar = k_star()**alpha - ((1 + g) * (1 + n) - (1 - delta)) * k_star()
    return cstar

def gamma(k):
    """Bounds on the feasible set of choices of
    consumption per effective worker.

    """
    lower = 0.0
    upper = (1 - delta) * k + k**alpha
    return (lower, upper)

def maximum(h, a, b):
    """Finds the maximum value of the function h on the interval [a,b]
    
    Returns a tuple containing the maximum value of h and its maximizer.
    
    """
    control, value = fminbound(lambda x: -h(x), a, b, full_output=True, disp=1)[0:2]
    return (-value, control)

def deterministicBellman(w, method='linear'):
    """The Bellman operator
    
    Arguments: 

        w:      A callable function
    
        method: Specifies the kind of interpolation as a string. Can be 
                one of:

                'linear'    - (Default) 1st order spline (i.e., linear 
                              interpolation).
                'quadratic' - 2nd order spline interpolation
                'cubic'     - 3rd order spline interpolation.
                'nearest'   - The nearest neighbor algorithm selects the 
                              value of the nearest point and does not 
                              consider the values of neighboring points 
                              at all, yielding a piecewise-constant 
                              interpolant. 

                Alternatively, can specify an integer indicating the 
                order of the spline interpolator to use. The default 
                method is 'linear'.

    """
    vals = []
    pols = []
    for state in grid:
        # current value function
        h = lambda control: u(state, control) + discount * w(capital(state, control))
        # find the value of the control that maximizes the current value function
        a, b = gamma(state)
        updatevector = maximum(h, a, b)
        # update the value function with maximum value of h
        vals.append(updatevector[0])
        # update the policy function with the maximizer of h
        pols.append(updatevector[1])
        
    return (interp1d(grid, vals, kind=method, bounds_error=False), \
            interp1d(grid, pols, kind=method, bounds_error=False))

# define model parameters
theta = 2.5
beta  = 0.99
g     = 0.005 
n     = 0.0025 
delta = 0.025 
alpha = 0.33

# define the discount factor
discount = beta * (1 + g)**(1 - theta) * (1 + n)

# grid of capital per effective worker (i.e., state variable!)
N = 250
kBar = 2 * k_star()
grid = np.linspace(kBar / N, kBar, N)

# Value function iteration algorithm
def initial_valueFunction(state):
    """Being smart about our initial guess can save
    substantial amount of computational time!

    """
    #return u(0, state)
    return u(0, c_star()) / (1 - discount)

current_valueFunction  = initial_valueFunction
current_policyFunction = lambda c: c_star()
error = 1
num_iter = 0

# actual tolerance is a function of discount factor!
tol = 0.01 * (1 - discount)

# create a new figure instance (we are going to plot value and policy iterates as we go!)
fig = plt.figure(figsize=(12,6))

# grid for plotting (different from state variable grid)
plot_grid = np.linspace(0, kBar, 1000)

# create two subplots, one for value function and one for policy function
ax1 = fig.add_subplot(121)
ax1.axhline(u(0, c_star()) / (1 - discount), 0, kBar, color='b', 
            linestyle='dashed', alpha=0.25)
ax1.set_xlim(0, kBar)
ax1.set_xlabel('$k_t$', fontsize=15)
ax1.set_ylabel('$V^*(k_t)$', fontsize=15, rotation='horizontal')
ax1.set_title('Optimal value function', weight='bold')

ax2 = fig.add_subplot(122)
ax2.axhline(c_star(), 0, kBar, color='g', linestyle='dashed', 
            alpha=0.25)
ax2.set_xlim(0, kBar)
ax2.set_xlabel('$k_t$', fontsize=15)
ax2.set_ylabel('$c(k_t)$', fontsize=15, rotation='horizontal')
ax2.set_title('Optimal consumption policy', weight='bold')

while True:
    next_valueFunction, next_policyFunction = deterministicBellman(current_valueFunction, method='cubic')
    error = np.max(np.abs(current_valueFunction(grid) - next_valueFunction(grid)))
    num_iter += 1
    if error < tol:
        final_valueFunction, final_policyFunction = next_valueFunction, next_policyFunction
        print "After", num_iter, "iterations, the final error is", error
        ax1.plot(plot_grid, final_valueFunction(plot_grid), 'b-', label='$V^*(k)$')
        ax2.plot(plot_grid, final_policyFunction(plot_grid), 'g-', label='$c_{VFI}(k)$')
        break
    else:
        current_valueFunction, current_policyFunction = next_valueFunction, next_policyFunction
        if num_iter % 50 == 0:
            print "After", num_iter, "iterations, the current error is", error
            ax1.plot(plot_grid, current_valueFunction(plot_grid), 'b--', alpha=0.25)
            ax2.plot(plot_grid, current_policyFunction(plot_grid), 'g--', alpha=0.25)

# add legends
ax1.legend(loc='best', frameon=False)
ax2.legend(loc='best', frameon=False)

fig.tight_layout()
plt.savefig('Graphics/Ramsey-value-policy-iterates.png')
plt.show()
