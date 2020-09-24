# %% md
# # Gold and USD inflation
# Using monthly data we will explore the relationship between gold prices and
# inflation.
#
# ### Inspirations
# - [The FRED® Blog: Is gold a good hedge against inflation?](https://fredblog.stlouisfed.org/2019/03/is-gold-a-good-hedge-against-inflation/)
#
# ### Definitions
# - Gold: The ICE Benchmark Administration Limited (IBA), Gold Fixing Price in
# the London Bullion Market, based in U.S. Dollars, retrieved from FRED,
# Federal Reserve Bank of St. Louis
# - CPI inflation: Consumer Price Index, seasonally adjusted monthly since 1947,
# , retrieved from FRED, Federal Reserve Bank of St. Louis
# - PCE inflation: Personal Consumption Expenditure, seasonally adjusted monthly
# since 1959, retrieved from FRED, Federal Reserve Bank of St. Louis
# - SYN inflation: Synthetic Inflation, normalized and averaged over CPI,
# CPI Core, PCE and PCE Core inflation measures, retrieved from FRED, Federal
# Reserve Bank of St. Louis
#
# ### Dependencies:
# - Python: datetime, fecon236, matplotlib, numpy, pandas, pandas_datareader,
# sklearn, statsmodels, sympy, seaborn
# - Written using Python 3.8.5, Atom 1.51, Hydrogen 2.14.4
#
# ### General To Do List:
# - Understand cluster analysis in greater detail
# - Review peer work
# %% md
# ## 0. Preamble: Code Setup and Function Definitions
# %% md
# ### Check if required modules are installed in the kernel; and if not install them
# %% codecell
import sys
import subprocess
import pkg_resources
required = {'datetime', 'fecon236', 'matplotlib', 'numpy', 'pandas',
            'pandas_datareader', 'sklearn', 'statsmodels', 'sympy', 'seaborn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing],
                           stdout=subprocess.DEVNULL)
# %% codecell
# fecon236 is a useful econometrics python module for access and using U.S.
# Federal Reserve and related data
import fecon236 as fe
fe.system.specs()
#  If a module is modified, automatically reload it:
%load_ext autoreload
%autoreload 2   # Use 0 to disable this feature.
# %% md
#  ### Import useful modules for data wrangling
# %% codecell
import numpy as np
import math
import datetime as dt
# %% codecell
# Will use sklearn and statsmodels for model development and testing
from sklearn import mixture
from sklearn.cluster import OPTICS, cluster_optics_dbscan
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
# from sklearn.datasets import make_blobs
import statsmodels.api as sm
# %% codecell
# Notebook display and formatting options
# Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option('display.notebook_repr_html', False)
# Alter Jupyter option for the pretty display of variables
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Import matplotlib and seaborn for plotting
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm   ### CAN WE GET RID OF THIS WITH SEABORN
#  Generate plots inside notebook, "inline" generates static png
%matplotlib inline
# Use seaborn for to make matplotlib pretty and set the default theme styles
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
# %% codecell
# Custom module containing a series of helper functions for processing
# and displaying the data
import sys
sys.path.append('../lib/')  # Append module directory so Jupyter can find it
import tools as yt
# %% codecell
# ### Variable Naming Schema:
# {series}_{type}_{period, optional}_{descriptor, optional}
# {series}
# dt    datetime
# in    inflation
# au    gold
#
# {type}
# idx   an index
# nom   nominal prices in USD
# rel   real (inflation adjusted) prices in USD
# lvl   levels e.g. prices, index values
# ppc   period on period percentage change
# fcd   fe module FRED codes
#
# {period}
# m     monthly
# y     yearly
#
# {descriptor}
# cln   column name
# var   variable identification
# %% md
# ## 1. Retrieve Data, Calculate Monthly & Annual Data for Nominal & Real Gold Prices
# %% md
# **Get gold and inflation rates, both as monthly frequency**
# %% codecell
# Notes: fecon236 uses median to resample (instead of say mean) and also
# replaces FRED empty data (marked with a ".") with data from previously
# occurring period; These adjustments will drive some small differences to
# the analysis on the FRED blog
# %% codecell
# Daily London AM gold fix, nominal USD, converted to monthly
dtau_nom_m = fe.monthly(fe.get('GOLDAMGBD228NLBM'))
# Daily London PM gold fix, nominal USD, converted to monthly
# au_nom_m = fe.get(fe.m4xau)
# %% codecell
# Inflation in use
in_fcd = fe.m4cpi      # FRED code 'CPIAUCSL'
# in_fcd = fe.m4pce    # FRED code 'PCEPI'
# Synthetic average of 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'CPILFESL'
# in_fcd = fe.m4infl
dtin_idx_m = fe.get (fe.m4cpi)  # Returns the index, not percentage change
# %% codecell
# Gold with USD inflation removed i.e. in real USD
# First, calculate rebased inflation index
in_basedate = '2020-08-01'              # The base date for our index
in_base = dtin_idx_m['Y'][in_basedate]
dtin_idx_rebased = fe.div(dtin_idx_m, in_base)
# %% codecell
# Find the first and last overlapping dates for the two data series where
# we are using values
dt_stp_lvl_m = max(fe.head(dtau_nom_m, 1).index[0],
                   fe.head(dtin_idx_m, 1).index[0])
dt_edp_lvl_m = min(fe.tail(dtau_nom_m, 1).index[0],
                   fe.tail(dtin_idx_m, 1).index[0])
# %% codecell
# Calculate the real adjusted for inflation gold price
dtau_rel_m = fe.div(dtau_nom_m.loc[dt_stp_lvl_m:dt_edp_lvl_m],
                    dtin_idx_rebased.loc[dt_stp_lvl_m:dt_edp_lvl_m])
# %% md
# **Determine period on period percentage changes**
# %% codecell
# Percentage calculation for month on month i.e. frequency = 1
freq = 1
dtin_ppc_m = fe.nona(fe.pcent(dtin_idx_m, freq))
dtau_nomppc_m = fe.nona(fe.pcent(dtau_nom_m, freq))
dtau_relppc_m = fe.nona(fe.pcent(dtau_rel_m, freq))
# %% codecell
# Find the first and last overlapping dates the monthly change data
dt_stp_ppc_m = max(fe.head(dtau_relppc_m, 1).index[0],
                   fe.head(dtin_ppc_m, 1).index[0])
dt_edp_ppc_m = min(fe.tail(dtau_relppc_m, 1).index[0],
                   fe.tail(dtin_ppc_m, 1).index[0])
# %% codecell
# Change percentage calculation to every 12 months for year on year changes
freq = 12
dtau_relppc_y = fe.nona(fe.pcent(dtau_rel_m, freq))
dtin_ppc_y = fe.nona(fe.pcent(dtin_idx_m, freq))
# %% codecell
# Find the first and last overlapping dates the yearly change data
dt_stp_ppc_y = max(fe.head(dtau_relppc_y, 1).index[0],
                   fe.head(dtin_ppc_y, 1).index[0])
dt_edp_ppc_y = min(fe.tail(dtau_relppc_y, 1).index[0],
                   fe.tail(dtin_ppc_y, 1).index[0])
# %% md
# ## 2. Plot and Review Time Series of Monthly Inflation and Gold Price Levels
# %% md
# ### Time series of the inflation index and gold prices in nominal and real terms
# %% codecell
# Create figure and axes
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 5));
# Plot chart on each axes
ax0.plot(dtin_idx_m.index, dtin_idx_m);
ax1.plot(dtau_nom_m.index, dtau_nom_m);
ax2.plot(dtau_rel_m.index, dtau_rel_m);
# Update titles
ax0.set_title('Inflation Index');
ax1.set_title('Gold Price (Nom. USD)');
ax2.set_title('Gold Price (Real USD)');
# %% md
# **2020-09-22 Results Discussion**
#
# 1. So we see a gradually increasing inflation index, a gold price in
# nominal terms that has a spike in ~1980, a second spike in ~2012, and a
# third spike in 2020; and lastly, looking at the gold price in real
# inflation adjusted terms, we see the three spikes again, but this time the
# spikes each roughly have the same maximum value at approx. $2,000
# (in current USD)
#
# 2. Taking a step back, one claim for gold is that it acts as a hedge against
# inflation i.e. as a nominal $100 decreasing in value due to inflation, the
# the value of a set amount of gold stays the same in real inflation adjusted
# currency terms. This relationship may hold out over very long time frames
# (i.e. centuries), but in the shorter term, the obvious volatility in the
# nominal and real prices indicates that there are many other factors at play
# in determining the gold price, and holding gold as purely an inflation hedge
# would be a poor investment decision.
#
# *Nonetheless, time to look at some analysis in greater detail to see if
# there is anything interesting in the data.*
# %% md
# ### Time series of the change in the inflation index and gold prices in nominal and real terms
# %% md
# **Monthly nominal gold price data**
# %% codecell
# Combine monthly inflation with nominal gold price data
dtinau_nomppc_m = pd.concat([dtin_ppc_m[dt_stp_ppc_m:dt_edp_ppc_m],
                             dtau_nomppc_m[dt_stp_ppc_m:dt_edp_ppc_m]], axis=1)
in_cln = 'Inflation'
au_nomppc_m_cln = 'MoM Nom. USD % Change'
dtinau_nomppc_m.columns = [in_cln, au_nomppc_m_cln]
dtinau_nomppc_m = pd.melt(dtinau_nomppc_m, ignore_index=False,
                          value_vars=[in_cln, au_nomppc_m_cln])
# %% codecell
# Update column names
inau_var_cln = 'Infl or Gold Price'
inau_m_val_cln = 'MoM % Change'
dtinau_nomppc_m.columns = [inau_var_cln, inau_m_val_cln]
# %% codecell
# Set palette for use
clrpalette = sns.color_palette(palette='husl', n_colors=2)
# %% codecell
# Plot and format axis labels
fcg = sns.relplot(data=dtinau_nomppc_m, x=dtinau_nomppc_m.index,
                  y=inau_m_val_cln, hue=inau_var_cln, alpha=0.5,
                  palette=clrpalette, kind='line');
yt.frmt_yaxislbls(fcg, fmt='{:.0f}', fmt0='{:.0%}', tickscle=1, tickscle0=0.01)
# %% md
# **Monthly real gold price data**
# %% codecell
# Combine monthly inflation with real gold price data
dtinau_relppc_m = pd.concat([dtin_ppc_m[dt_stp_ppc_m:dt_edp_ppc_m],
                             dtau_relppc_m[dt_stp_ppc_m:dt_edp_ppc_m]], axis=1)
au_relppc_m_cln = 'MoM Real USD % Change'
dtinau_relppc_m.columns = [in_cln, au_relppc_m_cln]
dtinau_relppc_m = pd.melt(dtinau_relppc_m, ignore_index=False,
                          value_vars=[in_cln, au_relppc_m_cln])
# %% codecell
# Update column names
dtinau_relppc_m.columns = [inau_var_cln, inau_m_val_cln]
# %% codecell
# Plot and format axis labels
fcg = sns.relplot(data=dtinau_relppc_m, x=dtinau_relppc_m.index,
                  y=inau_m_val_cln, hue=inau_var_cln, alpha=0.5,
                  palette=clrpalette, kind='line');
yt.frmt_yaxislbls(fcg, fmt='{:.0f}', fmt0='{:.0%}', tickscle=1, tickscle0=0.01)
# %% md
# **Yearly real gold price data**
# %% codecell
# Combine yearly inflation with real gold price data
dtinau_relppc_y = pd.concat([dtin_ppc_y[dt_stp_ppc_y:dt_edp_ppc_y],
                             dtau_relppc_y[dt_stp_ppc_y:dt_edp_ppc_y]], axis=1)
in_ppc_y_cln = 'Inflation'
au_relppc_y_cln = 'YoY Real USD % Change'
dtinau_relppc_y.columns = [in_ppc_y_cln, au_relppc_y_cln]
dtinau_relppc_y = pd.melt(dtinau_relppc_y, ignore_index=False,
                          value_vars=[in_ppc_y_cln, au_relppc_y_cln])
# %% codecell
# Update column names
inau_y_val_cln = 'YoY % Change'
dtinau_relppc_y.columns = [inau_var_cln, inau_y_val_cln]
# %% codecell
# Plot and format axis labels
fcg = sns.relplot(data=dtinau_relppc_y, x=dtinau_relppc_y.index,
                  y=inau_y_val_cln, hue=inau_var_cln, alpha=0.5,
                  palette=clrpalette, kind='line');
yt.frmt_yaxislbls(fcg, fmt='{:.0f}', fmt0='{:.0%}', tickscle=1, tickscle0=0.01)
# %% md
# **2020-09-23 Results Discussion**
#
# Difficult to identify any particular trends looking at the data this way.
# Likely easier to plot the data as scatter plots with inflation on the x-axis
# and change in gold price on the y-axis.
# %% md
# ## 3. Plot and Review Change in Inflation and Gold Price Levels
# ### 3.1 Monthly Data
# %% codecell
# Add the real data onto the inflation and noimnal data
dtinau_nomrelppc_m = dtinau_nomppc_m.append(dtinau_relppc_m.loc[dtinau_relppc_m[inau_var_cln] == au_relppc_m_cln])
# Pivot the table back into  multiple columns then melt and rename
inau_nomrelppc_m = dtinau_nomrelppc_m.pivot_table(values = inau_m_val_cln,
                                                  columns = inau_var_cln,
                                                  index = dtinau_nomrelppc_m.index)
inau_nomrelppc_m = pd.melt(inau_nomrelppc_m, ignore_index=False, id_vars=in_cln,
                          value_vars=[au_nomppc_m_cln, au_relppc_m_cln])
inau_nomrelppc_m.rename(columns={'value':inau_m_val_cln}, inplace=True)
# %% codecell
# Display the charts
fcg_nomrelppc_m = yt.snslmplot(data=inau_nomrelppc_m, xcol=in_cln,
                            ycol=inau_m_val_cln, yidcol=inau_var_cln, degree=1,
                            col_wrap=2, aspect=1.1)
plt_nomrelppc_m_title = ' vs. Inflation {} to {}'
plt_nomrelppc_m_title =  plt_nomrelppc_m_title.format(dt_stp_ppc_m.strftime("%b %Y"),
                                                      dt_edp_ppc_m.strftime("%b %Y"))
fcg_nomrelppc_m = fcg_nomrelppc_m.set_titles(col_template="{col_name}" +
                                             plt_nomrelppc_m_title)
# %% md
# **2020-09-22 Results Discussion** *(See Appendices for Statistics)*
#
# 1. For nominal prices, the low correlation coefficient (~0.15), poor ability
# of the model to explain movements (low R-squared and adjusted R-squareds of
# ~0.02) indicate that inflation explains only a small amount of the movement
# in nominal gold prices. The t-stat and p-values are erroneously high as
# changes in inflation will be included in the nominal price. To correct for
# this we need to remove inflation and use real prices.
# 2. For real prices, all model statistics show a weaker link as expected.
# %% md
# ### 3.2 Yearly Data
# %% codecell
# Pivot the dataframe from it's current melted shape
dtinau_relppc_y = dtinau_relppc_y.pivot_table(values=inau_y_val_cln,
                                              index=dtinau_relppc_y.index,
                                              columns=inau_var_cln)
# %% codecell
# Display the chart
fcg_relppc_y = yt.snslmplot(data=dtinau_relppc_y, xcol=in_ppc_y_cln,
                         ycol=au_relppc_y_cln, degree=1)
plt_relppc_y_title = 'Yearly Change in Gold Price (Real USD) vs. Inflation {} to {}'
plt_relppc_y_title =  plt_relppc_y_title.format(dt_stp_ppc_y.strftime("%b %Y"),
                                                dt_edp_ppc_y.strftime("%b %Y"))
for ax in fcg_relppc_y.axes.flat:
    fcg_relppc_y_ax = ax.set_title(plt_relppc_y_title)
# %% md
# **2020-09-22 Results Discussion** *(See Appendices for Statistics)*
#
# 1. A correlation coefficient of approx. 0.31 and a significant t-stat for the
# coefficient indicates that a yearly model is of better use than a monthly
# view. However, the r-squared and adjusted r-squareds are still small
# (approx. 0.1) indicating that the model is missing many other factors in
# determining the changes in gold price.
#
# 2. Of some interest is the group of data points in the upper right hand
# side of the chart. Does this indicate that gold prices change significantly
# when inflation is much higher than normal? Will explore higher order
# polynomial models and cluster analysis to see if there is anything of
# interest.
# %% md
# ### 3.3 Yearly Data with Higher Order Polynomials
# %% codecell
deg = 2     # Start at degree=2
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13.3, 10));
for axx in axs:
    for ax in axx:
        fcg = sns.regplot(data=dtinau_relppc_y, x='Inflation',
                          y=au_relppc_y_cln, order=deg, ax=ax)
        _ = ax.set_title('Polynomial Order: {0}'.format(deg))
        if deg != 4:
            _ = ax.set_xlabel('')
            _ = ax.set_ylabel('')
        deg += 1
# Provide some more white space to allow the plots to breathe
plt.subplots_adjust(wspace = 0.2)
plt.subplots_adjust(hspace = 0.2)
# %% codecell
# TODO: Custom tick label formatting
# %% md
# **2020-09-22 Results Discussion** *(See Appendices for Statistics)*
#
# Adding more features sees the statistics improve suggesting higher order
# polynomial models are a better fit (see number discussion below). However, a
# polynomial of order 2 or 4 are not intuitively obvious given their
# implications of both high inflation and high deflation leading to a high
# positive change in the gold price. Polynomials of order 1, 3 and 5 are
# more intuitively understandable, with high inflation leading to a high
# positive increase in gold prices, and high inflation leading to a high
# negative decrease in gold prices.
#
# The issue with higher order polynomials is one of over-fit, with the BIC
# number indicating that order 3 may be the sweet spot.
#
# Going with a order 3 polynomial model implies that gold prices do not
# really do anything during low to moderate inflation (perhaps up to ~8%
# just by eyeballing the data), but really takes off when inflation is high.
#
# **So, in 2020, are we in the foreseeable future likely to have high inflation
# reminiscent of the 1970's?** Unlikely in my view.
#
# Model statistics detail: The r-squared and adjusted r-squared's move higher
# to values of ~0.21 for `order=2`, to ~0.28 for `order=[3,4,5]`. The
# coefficient t-starts and p-values are significant for `order=[2,3]`, but not
# for 'order=[4,5]'. Reviewing the Bayesian Information Criterion (BIC),
# indicates that 'order=3' is the preferred model. The Akaike Information
# Criterion (AIC) indicates either 'order=[4,5]', with `order=3` a second
# choice.
# %% md
# ## 4 Cluster Analysis
# %% codecell
# Create NumPy array to hold data without the time series column
inau_relppc_y = np.column_stack((dtin_ppc_y['Y'][dt_stp_ppc_y:dt_edp_ppc_y], dtau_relppc_y['Y'][dt_stp_ppc_y:dt_edp_ppc_y]))
# %% md
# ### 4.1 Expectation-Maximization
# %% md
# The Expectation-Maximization Algorithm, or EM algorithm for short, is an
# approach for maximum likelihood estimation in the presence of latent
# variables.
# %% codecell
# Set up variables to determine the min and max plotting contour colours
# Start and stop take log values so a value of 2 is 100 (assuming base 10)
# Default choice
# start = 0
# stop = 2
# Based on knowing limits of Zln
# np.amin(Zln)
# np.amax(Zln)
start = math.log(10, 10)
stop = math.log(70, 10)
# %% codecell
# Define number of components to use
n_cpts = [1, 2, 3, 4, 6, 8, 10]
# Set up the figure for the subplots
colours = sns.color_palette('muted')
ncols = 2
nrows = math.ceil(len(n_cpts) / ncols)
fit = plt.figure(figsize=(15, 25))
# Run through each number of component to use
for axi in range(1, len(n_cpts) + 1):
    n_cpt = n_cpts[axi - 1]
    ax = plt.subplot(nrows, ncols, axi)
    # fit a GMM
    gmmmdl = mixture.GaussianMixture(n_components=n_cpt, covariance_type='full');
    _ = gmmmdl.fit(inau_relppc_y);
    # display predicted scores by the model as a contour plot
    xln = np.linspace(math.floor(min(inau_relppc_y[:, 0])),
                      math.ceil(max(inau_relppc_y[:, 0])))
    yln = np.linspace(math.floor(min(inau_relppc_y[:, 1])),
                      math.ceil(max(inau_relppc_y[:, 1])))
    Xln, Yln = np.meshgrid(xln, yln)
    XX = np.array([Xln.ravel(), Yln.ravel()]).T
    Zln = -gmmmdl.score_samples(XX)
    Zln = Zln.reshape(Xln.shape);
    # Create and display the plot
    CS = ax.contour(Xln, Yln, Zln, norm=LogNorm(vmin=1, vmax=100.0),
                     levels=np.logspace(start=0.3, stop=1.845, num=20, base=10));
    fcg = ax.scatter(inau_relppc_y[:, 0], inau_relppc_y[:, 1], .8);
    _ = ax.set_title('No. Components: {0}'.format(n_cpt));
    if deg == 10:
        CB = plt.colorbar(CS, shrink=0.8);
# Provide some more white space to allow the plots to breathe
plt.subplots_adjust(wspace = 0.1)
plt.subplots_adjust(hspace = 0.25)
# %% codecell
# TODO: Label axes
# TODO: Custom tick label formatting
# %% md
# **2020-09-22 Results Discussion**
#
# For 2 components EM essentially places a high likelihood
# around the cluster of data centred on [2.5, 0]. As we increase the number of
# components, EM starts to place more weight on the high-inflation,
# high-positive change in gold prices. However, no significant 'peak' forms
# in the upper-left, to me indicating that this method does not see much value
# in bisecting the data.
# %% md
# ### 4.2 K-Means
# %% codecell
# Compute the clustering with k-means
# Define number of clusters to use
n_clusters = [2, 3, 4, 5]
# Set up the figure for the subplots
colours = sns.color_palette('muted')
ncols = 2
nrows = math.ceil(len(n_clusters) / ncols)
fit = plt.figure(figsize=(13, 10))
# Run through each number of clusters to use
for axi in range(1, len(n_clusters) + 1):
    nci = n_clusters[axi - 1]
    ax = plt.subplot(nrows, ncols, axi)
    # Fit the model
    ppc_y_kmeans = KMeans(init='k-means++',
                          n_clusters=nci, n_init=10).fit(inau_relppc_y)
    kmeans_cluster_centers = ppc_y_kmeans.cluster_centers_
    kmeans_lbls = pairwise_distances_argmin(inau_relppc_y, kmeans_cluster_centers)
    # Plot the results
    for k, col in zip(range(nci), colours):
        my_members = kmeans_lbls == k
        cluster_center = kmeans_cluster_centers[k]
        fcg = ax.plot(inau_relppc_y[my_members, 0], inau_relppc_y[my_members, 1],
                       'w', markerfacecolor=col, marker='.')
        fcg = ax.plot(cluster_center[0], cluster_center[1], 'o',
                       markerfacecolor=col, markeredgecolor='k', markersize=6)

    _ = ax.set_title('No. of Clusters: {0}'.format(nci));
# Provide some more white space to allow the plots to breathe
plt.subplots_adjust(wspace = 0.1)
plt.subplots_adjust(hspace = 0.25)
# %% codecell
# TODO: Label axes
# TODO: Custom tick label formatting
# %% md
# **2020-09-15 Results Discussion**
#
# For `n_clusters = 2`, k-means splits the data essentially along
# horizontal axis, separating when the gold price change into two halves of
# when it is positive vs. negative. For `n_clusters = 3`, the data is further
# dissected along a horizontal line for gold change at approximately 50%.
# A similar trend occurs for `n_clusters = 4` with a further horizontal
# bisection. In summary, it is not obvious that horizontal clustering provides
# any insight into the gold vs. inflation relationship.
# %% md
# ### 4.3 OPTICS and DBSCAN
# %% codecell
# Define the fit parameters to use
clust = OPTICS(min_samples=20, xi=0.01, min_cluster_size=0.01)
# Run the fit
_ = clust.fit(inau_relppc_y)
space = np.arange(len(inau_relppc_y))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
# %% codecell
# Define the labels
eps = [0.1, 0.5, 1.0, 1.5, 2.0]
epslabels = []
for e in eps:
    _ = epslabels.append(cluster_optics_dbscan(reachability=clust.reachability_,
                                               core_distances=clust.core_distances_,
                                               ordering=clust.ordering_,
                                               eps=e));
# %% codecell
# Set up the figure for the reachability plot
colours = sns.color_palette('muted')
fig, ax = plt.subplots(figsize=(12, 8))
# Reachability plot
for klass, colour in zip(range(0, 10), colours):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    _ = ax.plot(Xk, Rk, ls='-', marker='.', color=colour, alpha=1)
_ = ax.plot(space[labels == -1], reachability[labels == -1], ls='', marker='.', color='k', alpha=0.2)
_ = ax.plot(space, np.full_like(space, 0.5, dtype=float), ls='-.', marker='', color='k', alpha=0.5)
_ = ax.plot(space, np.full_like(space, 1.0, dtype=float), ls='-', marker='', color='k', alpha=0.5)
_ = ax.plot(space, np.full_like(space, 2.0, dtype=float), ls='--', marker='', color='k', alpha=0.5)
_ = ax.set_ylim(0, 3)
_ = ax.set_ylabel('Reachability (epsilon distance)')
_ = ax.set_title('Reachability Plot')
# %% md
# **2020-09-24 Results Discussion**
#
# So with `OPTICS(min_samples=20, xi=0.01, min_cluster_size=0.01)` it appears
# that an `epsilon=1.5` may yield good results. Increasing `min_samples`
# increases epsilon (roughly the distance to other points within the same
# cluster) which broadens the criteria too far in my opinion. To reduce
# epsilon we can reduce the sample size. However, I think that reducing the
# sample size below approx. 20 reduces it too far loosing any relevance.
#
# Looking at different values for `xi=0.01, min_cluster_size=0.01` alter the
# location and cutoff for the reachability clusters (correct terminology?).
# Values of `0.01` seem to be the best compromise for this dataset.
#
# **For reference from the Wikipedia pages on [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
# and  [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm)**
#
# Good values of ε are where this plot shows an elbow; if epsilon is chosen
# much too small, a large part of the data will not be clustered; whereas for a
# too high value of epsilon, clusters will merge and the majority of objects
# will be in the same cluster. In general, small values of epsilon are
# preferable,and as a rule of thumb only a small fraction of points should be
# within this distance of each other.
#
# As a general rule for `min_samples` estimation, more points are generally
# required for noisy data and for data that contains many duplicates (as a rule
# of thumb, `min_samples` can be derived from the number of dimensions in the
# data set). As we've seen in earlier analysis, and as expected for typical
# financial trading data, there is a lot of noise in this dataset, hence
# `min_samples` should be large.
#
# Note, that there are different algorithms that try to detect the valleys by
# steepness, knee detection, or local maxima.
# %% codecell
# Set up the figure for the subplots
ncols = 2
nrows = math.ceil((len(eps) + 1)/ ncols)
fit = plt.figure(figsize=(13, 16))
# Run through each combination of the fit parameters
for axi in range(1, len(eps) + 2):
    ax = plt.subplot(nrows, ncols, axi)
    # Create and display the plot using OPTICS
    if axi == 1:
        for klass, colour in zip(range(0, 5), colours):
            inau_relppc_y_k = inau_relppc_y[clust.labels_ == klass]
            fcg = ax.plot(inau_relppc_y_k[:, 0], inau_relppc_y_k[:, 1], color=colour,
                         marker='o', ls='', alpha=0.3);
        fcg = ax.plot(inau_relppc_y[clust.labels_ == -1, 0],
                 inau_relppc_y[clust.labels_ == -1, 1], 'k+', alpha=0.1);
        title = ax.set_title('Automatic Clustering: OPTICS')
    # Create and display the plot using DBSCAN at eps
    else:
        epslabel = epslabels[axi - 2]
        for klass, colour in zip(range(0, 4), colours):
            inau_relppc_y_k = inau_relppc_y[epslabel == klass]
            fcg = ax.plot(inau_relppc_y_k[:, 0], inau_relppc_y_k[:, 1], color=colour,
                           marker='o', ls='', alpha=0.3)
        fcg = ax.plot(inau_relppc_y[epslabel == -1, 0],
                       inau_relppc_y[epslabel == -1, 1], 'k+', alpha=0.1);
        title = 'Clustering at {0:.2f} epsilon cut: DBSCAN'.format(eps[axi - 2])
        fcg = ax.set_title(title)
# Provide some more white space to allow the plots to breathe
plt.subplots_adjust(wspace = 0.1)
plt.subplots_adjust(hspace = 0.25)
# %% codecell
# TODO: Label axes
# TODO: Custom tick label formatting
# %% md
# **2020-09-24 Results Discussion**
#
# The `DBSCAN` seems to slign with `OPTICS` at for an epsilon of approx. 1.5.
#
# More broadly the methods appear to split the data horizontally similar to
# the EM and K-Means methods.
# %% md
# ### 4.4 Mean-Shift (MS)
# %% codecell
# Define the quantiles to use
quants = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
# %% codecell
# Set up the figure for the subplots
colours = sns.color_palette('muted')
ncols = 2
nrows = math.ceil((len(quants) + 1)/ ncols)
fig = plt.figure(figsize=(13, 20))
# Run through each combination of the fit parameters
for axi in range(1, len(quants) + 1):
    ax = plt.subplot(nrows, ncols, axi)
    quant = quants[axi - 1]
    # Calculate the MS
    bandwidth = estimate_bandwidth(inau_relppc_y, quantile=quant);
    apc_ms = MeanShift(bandwidth=bandwidth, bin_seeding=True);
    _ = apc_ms.fit(inau_relppc_y);
    lbls = apc_ms.labels_;
    cluster_centers = apc_ms.cluster_centers_;
    lbls_unique = np.unique(lbls);
    n_clusters_ = len(lbls_unique);
    # Plot result
    for k, colour in zip(range(n_clusters_), colours):
        my_members = lbls == k;
        cluster_center = cluster_centers[k];
        _ = ax.plot(inau_relppc_y[my_members, 0],
                       inau_relppc_y[my_members, 1],
                       color=colour, marker='o', ls='', alpha=0.3)
        _ = ax.plot(cluster_center[0], cluster_center[1], 'o',
                       markerfacecolor=colour, markeredgecolor='k', markersize=14)
    _ = ax.set_title('Quant of {} :: Est. clusters: {}'.format(quant, n_clusters_))
# %% codecell
# TODO: Label axes
# TODO: Custom tick label formatting
# Provide some more white space to allow the plots to breathe
plt.subplots_adjust(wspace = 0.1);
plt.subplots_adjust(hspace = 0.25);
# %% md
# 2020-09-15: And so the same story continues, clustering appears as horizontal
# bisections
# %% md
# ## Appendices
# ### A.1 Inflation vs. nominal gold prices, monthly, polynomial order = 1
# %% codecell
# Pivot the dataframe from it's current melted shape
dtinau_nomppc_m = dtinau_nomppc_m.pivot_table(values=inau_m_val_cln,
                                              index=dtinau_nomppc_m.index,
                                              columns=inau_var_cln)
# %% codecell
yt.dispmodel(dtinau_nomppc_m, degree=1)
# %% md
# ### A.2 Inflation vs. real gold prices, monthly, polynomial order = 1
# %% codecell
# Pivot the dataframe from it's current melted shape
dtinau_relppc_m = dtinau_relppc_m.pivot_table(values=inau_m_val_cln,
                                              index=dtinau_relppc_m.index,
                                              columns=inau_var_cln)
# %% codecell
yt.dispmodel(dtinau_relppc_m)
# %% md
# ### A.3 Inflation vs. real gold prices, yearly, polynomial order = 1
# %% codecell
yt.dispmodel(dtinau_relppc_y, degree=1)
# %% md
# ### A.4 Inflation vs. real gold prices, yearly, polynomial order = 2
# %% codecell
yt.dispmodel(dtinau_relppc_y, degree=2)
# %% md
# ### A.5 Inflation vs. real gold prices, yearly, polynomial order = 3
# %% codecell
yt.dispmodel(dtinau_relppc_y, degree=3)
# %% md
# ### A.6 Inflation vs. real gold prices, yearly, polynomial order = 4
# %% codecell
yt.dispmodel(dtinau_relppc_y, degree=4)
# %% md
# ### A.7 Inflation vs. real gold prices, yearly, polynomial order = 5
# %% codecell
yt.dispmodel(dtinau_relppc_y, degree=5)
# %% md
# ### End Of File
