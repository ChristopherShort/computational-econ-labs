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
     
# create a new model object
model = growth.SolowModel(cobb_douglas_output, marginal_product_capital)

# create a dictionary of steady state expressions
steady_state_funcs = {'k_star':analytic_k_star}

# pass it as an argument to the set_steady_state_functions method
model.steady_state.set_functions(steady_state_funcs)

# calibrate the model and compute steady state values
model.calibrate('GBR')

# create a new figure
fig = plt.figure(figsize=(8,6))

# plot the old Solow Diagram with new labels 
ax_old, y_old, actual_old, break_even_old = model.plot_solow_diagram(gridmax=20)
y_old.set_alpha(0.25)
y_old.set_linestyle('dashed')
y_old.set_label(r'$y_{old}$')

actual_old.set_alpha(0.25)
actual_old.set_linestyle('dashed')
actual_old.set_label(r'$i_{act, old}$')

# change the capital's share and recompute steady state values
model.args['alpha'] = 0.5
model.steady_state.set_values()

# create a new Solow Diagram
ax_new, y_new, actual_new, break_even_new = model.plot_solow_diagram(gridmax=20)
y_new.set_label(r'$y_{new}$')
actual_new.set_label(r'$i_{act, new}$')
ax_new.set_title(r'A rise in $\alpha$ leads to a rise in $k^*$!', fontsize=20, 
                 family='serif')

# add a legend
plt.legend(loc='best', frameon=False)

# display the figure
plt.show()
