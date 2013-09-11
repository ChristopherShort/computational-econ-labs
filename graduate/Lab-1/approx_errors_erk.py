import numpy as np
import matplotlib.pyplot as plt
from pyeconomics.models import growth

def cobb_douglas_output(t, k, params):
    """
    Cobb-Douglas production function.

    Arguments:

        t:      (array-like) Time.
        k:      (array-like) Capital (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        y: (array-like) Output (per person/ effective person)

    """
    # extract params
    alpha = params['alpha']
    
    # Cobb-Douglas technology
    y = k**alpha
    
    return y
    
def marginal_product_capital(t, k, params):
    """
    Marginal product of capital with Cobb-Douglas production technology.

    Arguments:

        t:      (array-like) Time.
        k:      (array-like) Capital (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        y_k: (array-like) Derivative of output with respect to capital, k.

    """
    # extract params
    alpha = params['alpha']

    return alpha * k**(alpha - 1)

def analytic_k_star(params): 
    """The steady-state level of capital stock per effective worker, k_bar, 
    in the Solow model is a function of the 5 exogenous parameters!
    
    """
    # extract params
    s     = params['s']
    n     = params['n']
    g     = params['g']
    alpha = params['alpha']
    delta = params['delta']
    
    return (s / (n + g + delta))**(1 / (1 - alpha))

def solow_analytic_solution(k0, t, params):
    """
    Computes the analytic solution for the Solow model with Cobb-Douglas
    production technology.
    
    Arguments:
        
        k0: (float) Initial value for capital (per person/effective person)     
        t:  (array-like) (T,) array of points at which the solution is 
            desired.
            
    Returns:
            
        analytic_traj: (array-like) (T,2) array representing the analytic 
                       solution trajectory.

    """
    # extract parameter values
    g     = params['g']
    n     = params['n']
    s     = params['s']
    alpha = params['alpha']
    delta = params['delta']
        
    # lambda governs the speed of convergence
    lmbda = (n + g + delta) * (1 - alpha)
        
    # analytic solution for Solow model at time t
    k_t   = (((s / (n + g + delta)) * (1 - np.exp(-lmbda * t)) + 
              k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))
        
    # combine into a (T, 2) array
    analytic_traj = np.hstack((t[:,np.newaxis], k_t[:,np.newaxis]))
    
    return analytic_traj
         
# create a new model object
params = {'s':0.1, 'n':0.02, 'g':0.02, 'delta':0.1, 'alpha':0.33}
model = growth.SolowModel(cobb_douglas_output, marginal_product_capital, params)

# create a dictionary of steady state expressions
steady_state_funcs = {'k_star':analytic_k_star}

# pass it as an argument to the set_steady_state_functions method
model.steady_state.set_functions(steady_state_funcs)
model.steady_state.set_values()

# solve the model using various methods
k0 = 6
h = 1.0
T = 200

forward_euler_traj = model.integrate(0, k0, T, h, 'forward_euler')
erk2_traj = model.integrate(0, k0, T, h, 'erk2')
erk3_traj = model.integrate(0, k0, T, h, 'erk3')
erk4_traj = model.integrate(0, k0, T, h, 'erk4')

grid = erk2_traj[:,0]
analytic_trajectory = solow_analytic_solution(k0, grid, model.args)

##### Approximation errors for RK methods #####
fig = plt.figure(figsize=(8,6))

# plot the forward Euler approximation error
benchmark_error = model.plot_approximation_error(forward_euler_traj, 
                                                 analytic_trajectory, 
                                                 log=True)[1]
benchmark_error.set_label('Forward Euler')
benchmark_error.set_marker('o')
benchmark_error.set_linestyle('none')

# plot the ERK2 approximation error
traj_error = model.plot_approximation_error(erk2_traj, 
                                            analytic_trajectory, 
                                            log=True)[1]
traj_error.set_label('ERK2')
#traj_error.set_color('r')
traj_error.set_marker('o')
traj_error.set_linestyle('none')

# plot the backward Euler approximation error
traj_error2 = model.plot_approximation_error(erk3_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error2.set_label('ERK3')
#traj_error2.set_color('r')
traj_error2.set_marker('o')
traj_error2.set_linestyle('none')

# plot the trapezoidal rule approximation error
traj_error3 = model.plot_approximation_error(erk4_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error3.set_label('ERK4')
#traj_error3.set_color('r')
traj_error3.set_marker('o')
traj_error3.set_linestyle('none')

# demarcate machine eps
plt.axhline(np.finfo('float').eps, color='k', ls='--', 
            label=r'Machine-$\epsilon$')
            
# Change the title and add a legend
plt.title('Approximation errors for explicit RK methods', 
          fontsize=20, family='serif')
plt.legend(loc='best', frameon=False, prop={'family':'serif'})

plt.savefig('graphics/solow-approximation-error-erk.png')
plt.savefig('graphics/solow-approximation-error-erk.pdf')
plt.show()

##### Compare convergence of RK4 with forward Euler #####

# solve the model using various methods
k0 = 6
h = 1.0
T = 200

forward_euler_traj = model.integrate(0, k0, T, h, 'forward_euler')
erk4_traj = model.integrate(0, k0, T, h, 'erk4')

grid = erk4_traj[:,0]
analytic_trajectory = solow_analytic_solution(k0, grid, model.args)

h = 0.1
forward_euler_traj_2 = model.integrate(0, k0, T, h, 'forward_euler')
erk4_traj_2 = model.integrate(0, k0, T, h, 'erk4')

grid = erk4_traj_2[:,0]
analytic_trajectory_2 = solow_analytic_solution(k0, grid, model.args)

fig = plt.figure(figsize=(8,6))

# plot the forward Euler approximation error
benchmark_error = model.plot_approximation_error(forward_euler_traj, 
                                                 analytic_trajectory, 
                                                 log=True)[1]
benchmark_error.set_label('Forward Euler, h=1.0')
benchmark_error.set_marker('o')
benchmark_error.set_linestyle('none')

benchmark_error2 = model.plot_approximation_error(forward_euler_traj_2, 
                                                  analytic_trajectory_2, 
                                                  log=True)[1]
benchmark_error2.set_label('Forward Euler, h=0.1')
benchmark_error2.set_color('b')
benchmark_error2.set_marker('^')
benchmark_error2.set_linestyle('none')

# plot the ERK4 approximation error
traj_error = model.plot_approximation_error(erk4_traj, 
                                            analytic_trajectory, 
                                            log=True)[1]
traj_error.set_label('ERK4, h=1.0')
traj_error.set_color('c')
traj_error.set_marker('o')
traj_error.set_linestyle('none')

traj_error2 = model.plot_approximation_error(erk4_traj_2, 
                                             analytic_trajectory_2, 
                                             log=True)[1]
traj_error2.set_label('ERK4, h=0.1')
traj_error.set_color('c')
traj_error2.set_marker('^')
traj_error2.set_linestyle('none')

# demarcate machine eps
plt.axhline(np.finfo('float').eps, color='k', ls='--', 
            label=r'Machine-$\epsilon$')
            
# Change the title and add a legend
plt.title(r'The difference between $\mathcal{O}(h)$ and $\mathcal{O}(h^4)$', 
          fontsize=20, family='serif')
plt.legend(loc='upper right', frameon=False, prop={'family':'serif'}) 
           #bbox_to_anchor=(1.45, 1.0))

plt.savefig('graphics/solow-convergence-erk4.png')#, bbox_inches='tight')
plt.savefig('graphics/solow-convergence-erk4.pdf')#, bbox_inches='tight')
plt.show()

##### Compare Forward Euler, RK5, and dopri5 #####

# solve the model using various methods
k0 = 6
h = 1.0
T = 200

forward_euler_traj = model.integrate(0, k0, T, h, 'forward_euler')
erk5_traj          = model.integrate(0, k0, T, h, 'erk5')
dopri5_traj        = model.integrate(0, k0, T, h, 'dopri5')

grid = erk5_traj[:,0]
analytic_trajectory = solow_analytic_solution(k0, grid, model.args)

fig = plt.figure(figsize=(8,6))

# plot the forward Euler approximation error
benchmark_error = model.plot_approximation_error(forward_euler_traj, 
                                                 analytic_trajectory, 
                                                 log=True)[1]
benchmark_error.set_label('Forward Euler')
benchmark_error.set_marker('o')
benchmark_error.set_linestyle('none')

# plot the ERK4 approximation error
traj_error = model.plot_approximation_error(erk5_traj, 
                                            analytic_trajectory, 
                                            log=True)[1]
traj_error.set_label('ERK5')
traj_error.set_color('c')
traj_error.set_marker('o')
traj_error.set_linestyle('none')

# plot the dopri5 approximation error
traj_error2 = model.plot_approximation_error(dopri5_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error2.set_label('dopri5')
traj_error.set_color('m')
traj_error2.set_marker('o')
traj_error2.set_linestyle('none')

# demarcate machine eps
plt.axhline(np.finfo('float').eps, color='k', ls='--', 
            label=r'Machine-$\epsilon$')
            
# Change the title and add a legend
plt.title(r'The benefits of adaptive step size', 
          fontsize=20, family='serif')
plt.legend(loc='best', frameon=False, prop={'family':'serif'})

plt.savefig('graphics/solow-erk5-dopri5.png', bbox_inches='tight')
plt.savefig('graphics/solow-erk5-dopri5.pdf', bbox_inches='tight')
plt.show()
