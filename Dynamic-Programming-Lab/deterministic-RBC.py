import numpy as np
from scipy import stats
from scipy.optimize import fminbound, fsolve
from scipy.interpolate import interp1d

def maximum(h, a, b):
    """Finds the maximum value of the function h on the interval [a,b]
    
    Returns a tuple containing the maximum value of h and its maximizer.
    
    """
    control, value = fminbound(lambda x: -h(x), a, b, 
                               full_output=True, disp=1)[0:2]
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
                'nearest'   - Nearest neighbor algorithm selects the 
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
        h = lambda control: u(state, control) + \
            discount * w(capital(state, control))
        # value of the control that maxes the current value function
        a, b = gamma(state)
        updatevector = maximum(h, a, b)
        # update the value function with maximum value of h
        vals.append(updatevector[0])
        # update the policy function with the maximizer of h
        pols.append(updatevector[1])
        
    return (interp1d(grid, vals, kind=method, bounds_error=False), \
            interp1d(grid, pols, kind=method, bounds_error=False))

def u(k, l):
    """Utility function"""
    return np.log(consumption(k, l)) + b * np.log(1 - l)
    
def output(k, l):
    """Output per effective person"""
    return k**alpha * l**(1 - alpha)

def real_wage(k, l):
    """labor is paid its marginal product"""
    return (1 - alpha) * k**alpha * l**(-alpha)

def consumption(k, l):
    """Intra-temporal FOC relating the two controls. This 
    equation allows us to avoid having to use two-dimensional
    interpolation.

    """
    return (real_wage(k, l) / b) * (1 - l)

def capital(k, l):
    """Evolution of capital per effective person"""
    y = output(k, l)
    c = consumption(k, l)
    kplus = (1 / ((1 + g) * (1 + n))) * ((1 - delta) * k + y - c)
    return kplus

def F1(k, l, c):
    """Implicit function describing equation of motion
    of capital per effective worker near steady state.

    """
    y = output(k, l)
    kplus = (1 / ((1 + g) * (1 + n))) * ((1 - delta) * k + y - c)
    return k - kplus

def F2(k, l, c):
    """Implicit function describing the consumption Euler
    equation near steady state.

    """
    r = alpha * (output(k, l) / k)
    return (1 / c) - (beta / (1 + g)) * (1 + r - delta) * (1 / c)

def F3(k, l, c):
    """Implicit FOC for consumption and labor supply decision
    near steady state.

    """
    return (c / (1 - l)) - (real_wage(k, l) / b)

def steadyStates(X):
    """Roots of this function are the steady state values."""
    out = [F1(X[0], X[1], X[2])]
    out.append(F2(X[0], X[1], X[2]))
    out.append(F3(X[0], X[1], X[2]))
    
    return out

# set values for model parameters
b     = 2.5
beta  = 0.99
g     = 0.005 
n     = 0.0025 
delta = 0.025 
alpha = 0.33

# determine the discount factor
discount = beta * (1 + n)

# Solve for the steady state values
k_star, l_star, c_star = fsolve(steadyStates, x0=(0.5, 0.5, 0.5))

print "Steady state value of k:", k_star
print "Steady state value of l:", l_star
print "Steady state value of c:", c_star

# grid of capital per effective worker (i.e., state variable!)
N = 250
kBar = 2.5 * k_star
#grid = np.logspace(np.log(kBar / N), np.log(kBar), N, base=np.exp(1))
grid = np.linspace(kBar / N, kBar, N)

def gamma(k):
    """Bounds on the feasible choice of labor supply."""
    lower = (1 - alpha) / (b + (1 - alpha))
        upper = 0.5 # kludge! This should be 1!
    return (lower, upper)

########## Start of the value function iteration algorithm ##########
def initial_valueFunction(state):
    """Being smart about our initial guess can save
    substantial amount of computational time!

    """
    #return u(0, state)
    return u(k_star, l_star) / (1 - discount)

current_value  = initial_valueFunction
error = 1
num_iter = 0

# actual tolerance is a function of discount factor!
tol = 0.01 * (1 - discount)

# we are going to plot value and policy iterates as we go!
fig = plt.figure(figsize=(18,6))

# grid for plotting (different from state variable grid)
plot_grid = np.linspace(0, kBar, 1000)

# create two subplots for value and policy functions
ax1 = fig.add_subplot(131)
ax1.axhline(u(k_star, l_star) / (1 - discount), 0, kBar, color='b', 
            linestyle='dashed', alpha=0.25)
ax1.set_xlim(0, kBar)
ax1.set_xlabel('$k_t$', fontsize=15)
ax1.set_ylabel('$V(k_t)$', fontsize=15, rotation='horizontal')
ax1.set_title('Optimal value function', weight='bold')

ax2 = fig.add_subplot(132)
ax2.axhline(l_star, 0, kBar, color='g', linestyle='dashed', 
            alpha=0.25)
ax2.set_xlim(0, kBar)
ax2.set_xlabel('$k_t$', fontsize=15)
ax2.set_ylabel('$l(k_t)$', fontsize=15, rotation='horizontal')
ax2.set_title('Optimal labor supply policy', weight='bold')

# optimal consumption policy
ax3 = fig.add_subplot(133)
ax3.axhline(consumption(k_star, l_star), 0, kBar, color='purple', 
            linestyle='dashed', alpha=0.25)
ax3.set_xlim(0, kBar)
ax3.set_xlabel('$k_t$', fontsize=15)
ax3.autoscale(axis='x', tight=True)
ax3.set_ylabel('$c(k_t)$', fontsize=15, rotation='horizontal')
ax3.set_title('Optimal consumption policy', weight='bold')

while True:
    next_value, next_policy = deterministicBellman(current_value, 
                                                   method='cubic')
    error = np.max(np.abs(current_value(grid) - next_value(grid)))
    num_iter += 1
    if error < tol:
        final_valueFunction  = next_value
        final_policyFunction = next_policy
        print "After", num_iter, "iterations, the final error is", error
        ax1.plot(plot_grid, final_valueFunction(plot_grid), 'b-', 
                 label='$V^*(k)$')
        ax2.plot(plot_grid, final_policyFunction(plot_grid), 'g-', 
                 label='$l_{VFI}(k)$')
        ax3.plot(plot_grid, 
                 consumption(plot_grid, final_policyFunction(plot_grid)), 
                 'purple', label='$c_{VFI}(k)$')
        break
    else:
        current_value, current_policy = next_value, next_policy
        if num_iter % 50 == 0:
            print "After", num_iter, "iterations, the current error is", error
            ax1.plot(plot_grid, current_value(plot_grid), 'b--', 
                     alpha=0.25)
            ax2.plot(plot_grid, current_policy(plot_grid), 'g--', 
                     alpha=0.25)
            ax3.plot(plot_grid, 
                     consumption(plot_grid, current_policy(plot_grid)), 
                     color='purple', linestyle='dashed', alpha=0.25)

# add legends
ax1.legend(loc='best', frameon=False)
ax2.legend(loc='best', frameon=False)
ax3.legend(loc='best', frameon=False)

fig.tight_layout()
plt.savefig('Graphics/RBC-value-policy-iterates.png')
plt.show()
