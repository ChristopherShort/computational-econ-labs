import numpy as np
import matplotlib.pyplot as plt

def output(k, t=0, params=None):
    """Output per effective worker.

    Arguments:

        k:      capital per effective worker
        params: a dictionary of parameter values for the Solow model.
        t:      time

    """
    # extract params
    alpha = params['alpha']

    if params.has_key('u') == False:
        return k**alpha
    elif params.has_key('u') == True:
        u = params['u']
        return (1 - u)**(1 - alpha) * k**alpha

def capital(k, t=0, params=None):
    """Equation of motion for capital per effective worker

    Arguments:

        k:      capital per effective worker
        params: a dictionary of parameter values for the Solow model.
        t:      time

    """
    # extract params
    delta = params['delta']
    g     = params['g']
    n     = params['n']
    s     = params['s']
    
    return  s * output(k, t, params) - (n + delta) * k

def k_bar(params):
    """Steady state value of capital per effective worker for Solow 
    model.

    Arguments:

        params: a dictionary of parameter values for the Solow model.

    """
    # extract params
    alpha = params['alpha']
    delta = params['delta']
    g     = params['g']
    n     = params['n']
    s     = params['s']

    if params.has_key('u') == False:
        return (s / (n + g + delta))**(1 / (1 - alpha))
    elif params.has_key('u') == True:
        u = params['u']
        return (1 - u) * (s / (n + g + delta))**(1 / (1 - alpha))

def y_bar(params):
    """Steady state value of output per effective worker for Solow 
    model.

    Arguments:

        params: a dictionary of parameter values for the Solow model.

    """
    return output(k_bar(params), params=params)

def c_bar(params):
    """Steady state value of consumption per effective worker for 
    Solow model.

    Arguments:

        params: a dictionary of parameter values for the Solow model.

    """
    # extract params
    s = params['s']
    
    return (1 - s) * y_bar(params)

def plot_solowDiagram(params, gridmax, N=1000, transparency=1.0):
    """Generates the classic Solow diagram.

    Arguments:

        params:       a dictionary of parameter values for the Solow 
                      model.
        gridmax:      max value of capital per effective worker to plot.
        N:            number of grid points to plot (default: N = 1000).
        transparency: controls transparency of the plot elements 
                      (default: transparency = 1.0)

    """
    # extract params
    alpha = params['alpha']
    delta = params['delta']
    g     = params['g']
    n     = params['n']
    s     = params['s']
    
    # grid of values for capital per worker
    grid = np.linspace(0, gridmax, N)

    # plot output, actual investment, and breakeven investment
    ax = plt.subplot(111)
    ax.plot(grid, output(grid, params=params), 'r', label='$f(k)$', 
            alpha=transparency)
    ax.plot(grid, s * output(grid, params=params), 'g', label='$sf(k)$', 
            alpha=transparency)
    ax.plot(grid, (n + g + delta) * grid, 'b', label='$i_{break}$', 
            alpha=transparency)

    # demarcate the steady state values
    ax.vlines(k_bar(params), 0, output(k_bar(params), params=params), 
              linestyles='dashed', colors='k', alpha=transparency)
    ax.hlines([s * output(k_bar(params), params=params), 
               output(k_bar(params), params=params)], 
               0, k_bar(params), linestyles='dashed', colors='k', 
               alpha=transparency)
    
    # axes, labels, title, legend, etc
    ax.set_xticks([k_bar(params)])
    ax.set_xticklabels([r'$\bar{k}$'])    
    ax.set_yticks([s * y_bar(params), y_bar(params)])
    ax.set_yticklabels([r'$\bar{i}$', r'$\bar{y}$']) 
    ax.set_xlabel('$k_t$\n' + 
                  r'$g=%0.3f,n=%0.3f,s=%.2f,\alpha=%.2f,\delta=%.2f$' \
                  %(g, n, s, alpha, delta), fontsize=15)
    ax.set_yticks((s * y_bar(params), y_bar(params)), 
                  ('$\bar{i}$', '$\bar{y}$')) 
    ax.set_ylabel('$y_t, i_t, c_t$', fontsize=15, rotation='horizontal')
    ax.set_title('Classic Solow diagram', weight='bold', fontsize=20)

    return ax
    
