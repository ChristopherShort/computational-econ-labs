import numpy as np
import matplotlib as mpl
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
params = {'s':0.1, 'n':0.025, 'g':0.025, 'delta':0.1, 'alpha':0.33}
model = growth.SolowModel(cobb_douglas_output, marginal_product_capital, params)

# create a dictionary of steady state expressions
steady_state_funcs = {'k_star':analytic_k_star}

# pass it as an argument to the set_steady_state_functions method
model.steady_state.set_functions(steady_state_funcs)
model.steady_state.set_values()

# solve the model using the forward Euler method
k0 = 2 * model.steady_state.values['k_star']

ab1_traj = model.integrate(0, k0, 1.0, T=100, integrator='forward_euler')

ab2_traj = model.integrate(0, k0, 1.0, T=100, integrator='ab2')

ab3_traj = model.integrate(0, k0, 1.0, T=100, integrator='ab3')

ab4_traj = model.integrate(0, k0, 1.0, T=100, integrator='ab4')

ab5_traj = model.integrate(0, k0, 1.0, T=100, integrator='ab5')

grid = ab2_traj[:,0]
analytic_trajectory = solow_analytic_solution(k0, grid, model.args)

fig = plt.figure(figsize=(8,6))

colors = mpl.cm.jet(np.linspace(0, 1, 5))

# plot the forward Euler approximation error
traj_error = model.plot_approximation_error(ab1_traj, 
                                            analytic_trajectory, 
                                            log=True)[1]
traj_error.set_label('s=1')
traj_error.set_marker('o')
traj_error.set_linestyle('none')

# plot the ab2 approximation error
traj_error1 = model.plot_approximation_error(ab2_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error1.set_label('s=2')
traj_error1.set_color(colors[1])
traj_error1.set_marker('o')
traj_error1.set_linestyle('none')

# plot the ab3 approximation error
traj_error2 = model.plot_approximation_error(ab3_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error2.set_label('s=3')
traj_error2.set_color(colors[2])
traj_error2.set_marker('o')
traj_error2.set_linestyle('none')

# plot the ab4 approximation error
traj_error3 = model.plot_approximation_error(ab4_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error3.set_label('s=4')
traj_error3.set_color(colors[3])
traj_error3.set_marker('o')
traj_error3.set_linestyle('none')

# plot the ab5 approximation error
traj_error4 = model.plot_approximation_error(ab5_traj, 
                                             analytic_trajectory, 
                                             log=True)[1]
traj_error4.set_label('s=5')
traj_error4.set_color(colors[4])
traj_error4.set_marker('o')
traj_error4.set_linestyle('none')

# demarcate machine eps
plt.axhline(np.finfo('float').eps, color='k', ls='--', 
            label=r'Machine-$\epsilon$')

# Change the title and add a legend
plt.ylim(1e-16, 1e0)
plt.title('Solow (1956) approximation error using\n' + 
          'Adams-Bashforth methods, $h=1.0$', fontsize=20, family='serif')
plt.legend(loc='best', frameon=False, prop={'family':'serif'})
plt.savefig('graphics/solow-approximation-error-ab-methods.pdf')
plt.show() 