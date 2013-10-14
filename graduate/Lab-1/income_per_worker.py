import numpy as np
import pandas as pd
import wbdata as wb
import matplotlib.pyplot as plt
import matplotlib as mpl

from pyeconomics.io import pwt

# Load the PWT data
data = pwt.load_pwt_data()

##### Divide countries into income groups #####

# Low income countries
LIC_countries = [country['id'] for country in \
                 wb.get_country(incomelevel="LIC", display=False)]

# Lower Middle income countries
LMC_countries = [country['id'] for country in \
                 wb.get_country(incomelevel="LMC", display=False)]

# Upper Middle income countries
UMC_countries = [country['id'] for country in \
                 wb.get_country(incomelevel="UMC", display=False)]

# High income countries
HIC_countries = [country['id'] for country in \
                 wb.get_country(incomelevel="HIC", display=False)]

##### Plot GDP per worker #####

# plot the level of technology
fig = plt.figure(figsize=(8,6))

# color scheme
colors = mpl.cm.jet(np.linspace(0, 1, 4), alpha=0.5)

for ctry in data.minor_axis:
    if ctry in LIC_countries:
        (data.rgdpna[ctry] / data.emp[ctry]).plot(color=colors[0], legend=False)
    elif ctry in LMC_countries:
        (data.rgdpna[ctry] / data.emp[ctry]).plot(color=colors[1], legend=False)
    elif ctry in UMC_countries:
        (data.rgdpna[ctry] / data.emp[ctry]).plot(color=colors[2], legend=False)
    elif ctry in HIC_countries:
        (data.rgdpna[ctry] / data.emp[ctry]).plot(color=colors[3], legend=False)
    
    val = (data.rgdpna[ctry] / data.emp[ctry]).ix[2010]
    if np.isnan(val) == False:
        plt.text(2010, val , ctry, fontsize=8)

# Axes, labels, title, etc
plt.yscale('log')
plt.ylabel(r'$\frac{Y}{L}$', family='serif', fontsize=15, rotation='horizontal')
plt.xlabel('Year', family='serif', fontsize=15)
plt.title(r'Real GDP per worker across income groups', family='serif', 
          fontsize=20)


plt.show()
