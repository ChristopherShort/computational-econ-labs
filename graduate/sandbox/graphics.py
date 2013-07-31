fig=plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# plot this trajectory
grid = np.linspace(0, 40, 1000)
ax.plot(grid, solow.get_analytic_solution(0.5, grid), 'r-', label='$k(t)$')

k0  = solow.get_analytic_solution(0.5, 10)
k_E = k0 + (30 - 10) * capital(10, k0, model.args)

ax.plot(np.linspace(8, 30, 100), k0 + (np.linspace(8, 30, 100) - 10) * capital(10, k0, model.args))
ax.vlines(x=[10, 30], ymin=0, ymax=[k0, k_E], linestyles='dashed', color='grey', alpha=0.5)
ax.hlines(y=k0, xmin=10, xmax=30, color='k')
ax.vlines(x=30, ymin=k0, ymax=k_E, color='k')

# remove the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# hide the top and right ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# axes, labels, title, etc
ax.set_xticks([10, 30])
ax.set_xticklabels(['$t_n$', '$t_{n+1}$'])
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=15)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylim(0, 3.5)

# annotate
ax.text(9.5, 1.075 * k0, '$k_n$')
ax.text(30, k_E, '$k_{n+1}^E$')
ax.text(17.5, 1.075 * k0, '$h=(t_{n+1} - t_n)$')
ax.text(30, 0.7 * k_E, '$hf(t_n, k_n)$')
        
ax.set_title('The forward Euler method', fontsize=20, family='serif')
ax.legend(loc='best', frameon=False)

plt.savefig('graphics/forward-euler.png')
plt.show()

fig=plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# plot this trajectory
grid = np.linspace(0, 40, 1000)
ax.plot(grid, solow.get_analytic_solution(0.5, grid), 'r-', label='$k(t)$')

k0  = solow.get_analytic_solution(0.5, 10)
k_B = optimize.bisect(lambda k: k - (k0 + (30 - 10) * capital(30, k, model.args)), k0, 5)

ax.plot(np.linspace(8, 30, 100), k0 + (np.linspace(8, 30, 100) - 10) * capital(10, k_B, model.args), 'g-')
ax.vlines(x=[10, 30], ymin=0, ymax=[k0, k_B], linestyles='dashed', color='grey', alpha=0.5)
ax.hlines(y=k0, xmin=10, xmax=30, color='k')
ax.vlines(x=30, ymin=k0, ymax=k_B, color='k')

# remove the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# hide the top and right ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# axes, labels, title, etc
ax.set_xticks([10, 30])
ax.set_xticklabels(['$t_n$', '$t_{n+1}$'])
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=15)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylim(0, 3.5)

# annotate
ax.text(9.5, 1.075 * k0, '$k_n$')
ax.text(30, k_B, '$k_{n+1}^B$')
ax.text(17.5, 1.075 * k0, '$h=(t_{n+1} - t_n)$')
ax.text(30, 0.75 * k_B, '$hf(t_n, k_{n+1})$')
        
ax.set_title('The backward Euler method', fontsize=20, family='serif')
ax.legend(loc='best', frameon=False)

plt.savefig('graphics/backward-euler.png')
plt.show()

fig=plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# time interval end points
t0 = 10
t1 = 30

# plot analytic trajectory
grid1 = np.linspace(0, t0 + t1, 1000)
ax.plot(grid, solow_analytic_solution(0.5, grid, solow.args)[:,1], 'r-', label='$k(t)$')

trapezoidal_rule = lambda k: k - (k0 + 0.5 * (t1 - t0) * (solow.capital(t0, k0, solow.args) + solow.capital(t1, k, model.args)))

k0  = solow_analytic_solution(0.5, np.array([t0]), solow.args)[0,1]
k1 = optimize.bisect(trapezoidal_rule, k0, 5)

grid2 = np.linspace(8, 30, 100)
ax.plot(grid2, k0 + 0.5 * (grid2 - t0) * (solow.capital(t0, k0, solow.args) + solow.capital(t1, k1, model.args)), 'y-')
ax.vlines(x=[t0, t1], ymin=0, ymax=[k0, k1], linestyles='dashed', color='grey', alpha=0.5)
ax.hlines(y=k0, xmin=t0, xmax=t1, color='k')
ax.vlines(x=t1, ymin=k0, ymax=k1, color='k')

# remove the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# hide the top and right ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# axes, labels, title, etc
ax.set_xticks([t0, t1])
ax.set_xticklabels(['$t_n$', '$t_{n+1}$'])
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=15)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylim(0, 3.5)

# annotate
ax.text(9.5, 1.075 * k0, '$k_n$')
ax.text(t1, k1, '$k_{n+1}^T$')
ax.text(17.5, 1.075 * k0, '$h=(t_{n+1} - t_n)$')
ax.text(t1, 0.75 * k1, r'$\frac{1}{2}h\left[f(t_n, k_n) + f(t_{n+1}, k_{n+1})\right]$')
        
ax.set_title('The trapezoidal rule', fontsize=20, family='serif')
ax.legend(loc='best', frameon=False)

plt.savefig('graphics/trapezoidal-rule.png')
plt.show()

fig=plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# time interval end points
t0 = 10
t1 = 30

# plot analytic trajectory
grid = np.linspace(0, t0 + t1, 4)
ax.plot(grid, solow_analytic_solution(0.5, grid, solow.args)[:,1], 'ro-', label='$k(t)$')

# remove the right and top spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# hide the top and right ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# axes, labels, title, etc
ax.set_xlim(5, 35)
ax.set_xticks(np.linspace(0, t0 + t1, 4))
ax.set_xticklabels(['$t_{n-1}$', '$t_{n}$', '$t_{n+1}$', '$t_{n+2}$'])
ax.set_ylabel('$k(t)$', rotation='horizontal', fontsize=15)
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_ylim(0, 3.5)
        
ax.set_title('Linear interpolation', fontsize=20, family='serif')
ax.legend(loc='best', frameon=False)

plt.savefig('graphics/linear-interpolation.png')
plt.show()