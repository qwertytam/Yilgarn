# %% md
# # Gold and USD inflation
# Using monthly data we will explore the relationship between gold prices and
# inflation.
#
# ### Inspirations
# - [The FRED® Blog: Is gold a good hedge against inflation?]
# (https://fredblog.stlouisfed.org/2019/03/is-gold-a-good-hedge-against-inflation/)
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
# - Python: matplotlib, pandas, fecon236, statsmodels, sympy, pandas_datareader
# - Written using Python 3.8.5, Atom 1.51, Hydrogen 2.14.4

# %% md
# ## 0. Preamble / Code Setup
# %% codecell
# Check if required modules are installed in the kernel; and if not install them
import sys
import subprocess
import pkg_resources

required = {'fecon236', 'pandas', 'numpy', 'sklearn', 'statsmodels', 'sympy',
            'pandas_datareader', 'matplotlib', 'datetime'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(missing)
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing],
                           stdout=subprocess.DEVNULL)
    print('install process finished')
else:
    print('no install required')
# %% codecell
# fecon236 is a useful econometrics python module for access and using U.S.
# Federal Reserve and related data
import fecon236 as fe
fe.system.specs()
pwd = fe.system.getpwd()
print(" :: $pwd", pwd)

#  If a module is modified, automatically reload it:
%load_ext autoreload
%autoreload 2   # Use 0 to disable this feature.
#  Notebook DISPLAY options:
# Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option('display.notebook_repr_html', False)

from matplotlib import pyplot as plt
#  Generate PLOTS inside notebook, "inline" generates static png:
%matplotlib inline
# "notebook" argument allows interactive zoom and resize.

import numpy as np
import math
import datetime as dt

# Will use sklearn and statsmodels for regression model testing
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# These are the "Tableau 20" colors as RGB for later use
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format
# matplotlib accepts.
for i in range(len(tableau20)):
    rescalef = 255.
    r, g, b = tableau20[i]
    tableau20[i] = (r / rescalef, g / rescalef, b / rescalef)
# %% md
# ## 1. Define Custom Functions
# ### Custom Plotting Function to Make Charts *Pretty*
# %% codecell
def plotfn(x, y, poly1d_fn, title: str=''):
    """Draws plot from data input and defined fit function

    Parameters
    ----------
    x : panda series
        x axis data
    y : panda series
        y axis data
    poly1d_fun : numpy poly1d
        the polynomial definition e.g. x**2 + 2*x + 3
    title : str, optional
        title to include on the chart

    Returns
    -------
    Null
    """
    # Calculate ranges and intervals for chart display
    xmin = int(math.floor(min(x)))
    xmax = int(math.ceil(max(x)))
    xrange = xmax - xmin
    xnumticks = 8
    xtickintvl = round(xrange / xnumticks * 2, 0) / 2

    ymin = int(math.floor(min(y)))
    ymax = int(math.ceil(max(y)))
    yrange = ymax - ymin
    ynumticks = 10
    ytickintvl = int(math.ceil(yrange / ynumticks))

    # Plot aspect ratio of ~1.33x
    plt.figure(figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Limit the range of the plot to only where the data is to avoid
    # unnecessary whitespace
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    # Make sure axis ticks are large enough to be easily read
    # You don't want your viewers squinting to read your plot
    tickfntsize = 14
    plt.yticks(range(ymin, ymax + ytickintvl, ytickintvl),
               [str(x) + "%" for x in range(ymin, ymax + ytickintvl, ytickintvl)],
               fontsize=tickfntsize)
    plt.xticks(np.arange(xmin, xmax + xtickintvl, xtickintvl),
               [str(x) + "%" for x in np.arange(xmin, xmax + xtickintvl, xtickintvl)],
               fontsize=tickfntsize)

    # Provide tick lines across the plot to help viewers trace along
    # the axis ticks; lines are light and small so they don't obscure the
    # primary data lines.
    xarange = np.arange(xmin, xmax + 0.001)
    for yt in range(ymin, ymax + ytickintvl, ytickintvl):
        plt.plot(xarange, [yt] * len(xarange), "--",
                 lw=0.5, color="black", alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines
    # we just plotted.
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=True, left=False, right=False,
                    labelleft=True)

    # Add title
    plt.text(xmin + xrange / 2, ymax + yrange * 0.05, title,
             fontsize=17, ha='center')

    # Plot the two data series
    result = plt.plot(x, y, color=tableau20[1], marker='o', ls='')
    result = plt.plot(x, poly1d_fn(x), color=tableau20[2], ls='-')
    return
# %% md
# ### Function to Define Polynomial Model
# %% codecell
def defnmodel(x, y, degree: int=1):
    """Defines a polynomial model for data x and y of order degree using
    ordinary least squares regression based

    Parameters
    ----------
    x : panda series
        x axis data
    y : panda series
        y axis data
    degree : int, optional
        the polynomial degree definition e.g. 2 for a quadractic polynomial

    Returns
    -------
    xp : numpy array
        array of degree columns containing the polynomial features e.g. for a 2
        degree polynomial, features are [1, a, b, a^2, ab, b^2]
    model : sm ols model
        orindary least squares regression model produced by statsmodels
    poly1d_fn : numpy poly1d
        the polynomial definition e.g. x**2 + 2*x + 3
    """
    polyFeatures = PolynomialFeatures(degree) # Define the polynomial

    # Reshape data from 1 by n to n by 1
    xarray = np.array(x)
    xarray = xarray[:, np.newaxis]

    # Calculate polynomials for x
    xp = polyFeatures.fit_transform(xarray)
    # Reshape y from 1 by n to n by 1
    yarray = np.array(y)
    yarray = yarray[:, np.newaxis]

    # Calculate the model and predictions
    model = sm.OLS(yarray, xp).fit()
    coef = model.params.tolist()    # Model coefficients
    coef.reverse()                  # Reverse as poly1d takes in decline order
    poly1d_fn = np.poly1d(coef)     # Create function from coefficients

    return xp, model, poly1d_fn
# %% md
# ### Roll-up Function to Display Results
# %% codecell
def displayresults(x, y, degree: int=1, title: str=''):
    """Displays summary statistics, regression results, and plot of data
    versus polynomial model

    Parameters
    ----------
    x : panda series
        x axis data
    y : panda series
        y axis data
    degree : int, optional
        the polynomial degree definition e.g. 2 for a quadractic polynomial
    title : str, optional
        title to include on the chart

    Returns
    -------
    Null
    """
    # Display the summary stats and chart
    xp, model, poly1d_fn = defnmodel(x, y, degree)
    print(" ::  FIRST variable (y):")
    print(y.describe(), '\n')
    print(" ::  SECOND variable (x):")
    print(x.describe(), '\n')
    print(model.summary())
    plotfn(x, y, poly1d_fn, title)
    return
# %% md
# ## 1. Retrieve Data, Determine Appropiate Start and End Dates for Analysis
# %% codecell
# Get gold and inflation rates, both as monthly frequency
# Notes: fecon236 uses median to resample (instead of say mean) and also
# replaces FRED empty data (marked with a ".") with data from previously
# occuring period; These adjustments will drive some small differences to
# the analysis on th FRED blog
# Daily London AM gold fix, nominal USD, converted to monthly
gold_usdnom = fe.monthly(fe.get('GOLDAMGBD228NLBM'))
# Daily London PM gold fix, nominal USD, converted to monthly
# gold_usdnom = fe.get(fe.m4xau)
# Percentage calcualtion for month on month i.e. frequency = 1
freq = 1
gold_usdnom_pc = fe.nona(fe.pcent(gold_usdnom, freq))
# %% codecell
# Inflation in use
infcode = fe.m4cpi      # FRED code 'CPIAUCSL'
# infcode = fe.m4pce    # FRED code 'PCEPI'
# Synthetic average of 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'CPILFESL'
# infcode = fe.m4infl
infidx = fe.get (fe.m4cpi)        # Returns the index, not percentage change
inf_pc = fe.nona(fe.pcent(infidx, freq))
# %% codecell
# Gold with USD inflation removed i.e. in real USD
# First, calculate rebased inflation index
inf_basedate = '2020-08-01'              # The base date for our index
inf_base = infidx['Y'][inf_basedate]
infidx_rebased = fe.div(infidx, inf_base)
# %% codecell
# Find the first and last overlapping dates for the two data series where
# we are using values
startv = max(fe.head(gold_usdnom, 1).index[0], fe.head(infidx, 1).index[0])
endv = min(fe.tail(gold_usdnom, 1).index[0], fe.tail(infidx, 1).index[0])
# %% codecell
# Calculate the real gold price
gold_usdrl = fe.div(gold_usdnom.loc[startv:endv],
                    infidx_rebased.loc[startv:endv])
gold_usdrl_pc = fe.nona(fe.pcent(gold_usdrl, freq))
# %% codecell
# Find the first and last overlapping dates for the two data series where we
# are using month on month percentage change
startpc = max(fe.head(gold_usdrl_pc, 1).index[0], fe.head(inf_pc, 1).index[0])
endpc = min(fe.tail(gold_usdrl_pc, 1).index[0], fe.tail(inf_pc, 1).index[0])
# %% md
# ## 2. Look at the Correlation of Month on Month Change in Inflation and Nominal Gold Prices
# %% codecell
x = inf_pc['Y'][startpc:endpc]
y = gold_usdnom_pc['Y'][startpc:endpc]
title = 'MoM Change in Nominal Gold Price vs Inflation {} to {}'
title = title.format(startpc.strftime("%b %Y"), endpc.strftime("%b %Y"))
displayresults(x, y, 1, title)
# %% md
# 2020-09-15: The regression analysis shows a strong relationship
# (t-stat 3.896), however we need to remove the inflation contained in the
# nominal gold price
# %% md
# ## 3. Look at the Correlation of Month on Month Change in Inflation and Real Gold Prices
# %% codecell
x = inf_pc['Y'][startpc:endpc]
y = gold_usdrl_pc['Y'][startpc:endpc]
title = 'MoM Change in Real Gold Price vs Inflation {} to {}'
title = title.format(startpc.strftime("%b %Y"), endpc.strftime("%b %Y"))
displayresults(x, y, 1, title)
# %% md
# 2020-09-15: The regression analysis shows a relationship
# (t-stat 2.245), with a 1% increase in inflation having a 1.3573% increase in
# the real price of gold. However, the adj. r-squared is only 0.006, indicating
# there are many other factors at play that influence the change in gold prices.
# %% md
# ## 4. Look at the Correlation of Year on Year Change in Inflation and Real Gold Prices
# %% codecell
# Change percentage calcualtion to every 12 months
freq = 12
gold_usdrl_apc = fe.nona(fe.pcent(gold_usdrl, freq))
inf_apc = fe.nona(fe.pcent(infidx, freq))
# %% codecell
# Find the first and last overlapping dates for the two data series
startapc = max(fe.head(gold_usdrl_apc, 1).index[0], fe.head(inf_apc, 1).index[0])
endapc = min(fe.tail(gold_usdrl_apc, 1).index[0], fe.tail(inf_apc, 1).index[0])
# %% codecell
# Show same analysis as above
x = inf_apc['Y'][startapc:endapc]
y = gold_usdrl_apc['Y'][startapc:endapc]
title = 'YoY Change in Real Gold Price vs Inflation {} to {}'
title = title.format(startapc.strftime("%b %Y"), endapc.strftime("%b %Y"))
displayresults(x, y, 1, title)
# %% md
# 2020-09-15: The regression analysis shows a relationship
# (t-stat 8.119), with a 1% increase in inflation having a 2.5554% increase in
# the real price of gold; the adj. r-squared has increased slightly to 0.095, so
# still many other factors at play in determing the gold price
#
# Another way to consider this is, yes gold may hedge inflation, but with so
# many other (so far unknown) factors impacting the price of gold, there is lot
# of risk in using gold to purely hedge inflation
#
# Potentially there is greater movement in the gold price when the inflation
# change is > ~8%
# %% md
# ## 5. Alternative Models for Year on Year Change in Inflation and Real prices
# ### First, lets explore a polynomial model
# %% codecell
x = inf_apc['Y'][startapc:endapc]
y = gold_usdrl_apc['Y'][startapc:endapc]
displayresults(x, y, 2, title)
# %% md
# 2020-09-15: For `degree = 2` We show a stronger relationship (higher t-stats
# and adj. r-squared of 0.212), however it is not clear why a quadractic
# equation is an appropiate relationship between inflation and gold prices
# %% codecell
displayresults(x, y, 3, title)
# %% md
# 2020-09-15: For `degree = 3`, relationship is weaker and not interesting
# %% codecell
displayresults(x, y, 4, title)
# %% md
# 2020-09-15: For `degree = 4`, relationship is even weaker and not interesting
# %% codecell
displayresults(x, y, 5, title)
# %% md
# 2020-09-15: For `degree = 5`, now relationships are unidentifiable
