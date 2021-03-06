'''
_______________|  tools.py :: Assorted tools for analysis of gold

CHANGE LOG  For LATEST version, see https://github.com/qwertytam/Yilgarn/blob/master/nb/usdi-xau.ipynb
2020-09-22  Initial version
'''
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# %% codecell
def getaxes(fcg):
    """Returns list of matpotlib axes objects

    Parameters
    ----------
    fcg : seaborn FaceGrid
        A FaceGrid object that contains the axes

    Returns
    -------
    [matplotlib.axes._subplots.AxesSubplot]
        A list of axes objects
    """

    axobjs = fcg.axes
    # If already a subplot, then change to a list
    if type(axobjs) == plt.Subplot:
        axobjs = [axobjs]
    # If only one plot, then fcg.axes is a list of lists, so addressable
    # via fcg.axes[0][0]
    elif type(axobjs[0]) == np.ndarray:
        axobjs = axobjs[0]
    # If multiple plots, then it is a list of objects, so addressable via
    # fcg.axes[0]
    else:
        axobjs = axobjs

    return axobjs

def fmtticks(fmt, fmt0, ticks, tickscle, tickscle0):
    """Formats x and y axis tick labels for a seaborn plot

    Parameters
    ----------
    fmt : str
        The formatting string to use for all tick labels
    fmt0 : str
        The formatting string to use for only the first tick label
    ticks : numpy.ndarray
        1D array of tick labels to format
    tickscle  : float, optional
        Scale factor to apply to all tick labels e.g. if format string
        is for a %, and the labels are in the range 0 to 100,
        then numscle = 0.01 to change the range to 0 to 1
    tickscle0 : float, optional
        Scale factor to apply to only the first tick label

    Returns
    -------
    list of str
        List of formatted tick labels in string format
    """

    tick0 = ticks[0]
    lbls = [fmt.format(tick * tickscle) for tick in ticks]
    lbl0 = fmt0.format(tick0 * tickscle0)
    lbls[0] = lbl0

    return lbls

def frmt_xaxislbls(fcg, fmt: str='{:.0}', fmt0: str=':.0%',
                 tickscle: float=1.0, tickscle0: float=1.0):
    """Formats x axis tick labels for each axis on a seaborn plot

    Parameters
    ----------
    fcg : seaborn FacetGrid
        A FacetGrid that contains the axis labels for formatting
    fmt : str, optional
        The formatting string to use for all tick labels
    fmt0 : str, optional
        The formatting string to use for only the first tick label
    numscle : float, optional
        Scale factor to apply to all tick labels e.g. if format string
        is for a %, and the labels are in the range 0 to 100,
        then numscle = 0.01 to change the range to 0 to 1
    numscle0 : float, optional
        Scale factor to apply to only the first tick label

    Returns
    -------
    Null
    """

    axobjs = getaxes(fcg)
    for ax in axobjs:
        ticks = ax.get_xticks()
        lbls = fmtticks(fmt, fmt0, ticks, tickscle, tickscle0)
        ax.set_xticks(ax.get_xticks().tolist()) # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - IT IS A BUG FROM MATPLOTLIB 3.3.1        
        fcg.set_xticklabels(lbls)

    return

def frmt_yaxislbls(fcg, fmt: str='{:.0}', fmt0: str=':.0%',
                 tickscle: float=1.0, tickscle0: float=1.0):
    """Formats y axis tick labels for each axis on a seaborn plot

    Parameters
    ----------
    fcg : seaborn FacetGrid
        A FacetGrid that contains the axis labels for formatting
    fmt : str, optional
        The formatting string to use for all tick labels
    fmt0 : str, optional
        The formatting string to use for only the first tick label
    numscle : float, optional
        Scale factor to apply to all tick labels e.g. if format string
        is for a %, and the labels are in the range 0 to 100,
        then numscle = 0.01 to change the range to 0 to 1
    numscle0 : float, optional
        Scale factor to apply to only the first tick label

    Returns
    -------
    Null
    """

    axobjs = getaxes(fcg)
    for ax in axobjs:
        ticks = ax.get_yticks()
        lbls = fmtticks(fmt, fmt0, ticks, tickscle, tickscle0)
        ax.set_yticks(ax.get_yticks().tolist()) # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - IT IS A BUG FROM MATPLOTLIB 3.3.1
        fcg.set_yticklabels(lbls)

    return

def snslmplot(data: pd.core.frame.DataFrame, xcol: str, ycol: str,
              yidcol: str=None, degree: int=1, col_wrap: int=None,
              title: str=None, axistitles: str=None, aspect=1.333):
    """Draws plot from data input with polynomial of order degree

    Parameters
    ----------
    data : panda DataFrame
        Panda melted data frame containing the data, m rows by 3 columns
        The three columns are identified by the xcol, idcol, datacol function
        parameters - see below
    xcol : str
        The name of the DataFrame column that contains data for the independent
        variable i.e. x
    ycol : str
        The name of the DataFrame colum that contains the y data
    yidcol : str, optional
        The name of the DataFrame column that contains data identifying the y
        data to select. For example, if the data has been melted from two
        variables 'y1' and 'y2', this column would contain either 'y1' or 'y2'
        to identify which rows pertain to the relevant y variable
    degree : int, optional
        The polynomial degree definition e.g. 2 for a quadratic polynomial
    col_wrap : int, optional
        The number of facets to display per row i.e. wrap on
    title : str, optional
        Title to include on the chart; can also be a list of strings
    axistitles : str, optional
        Title for each of the titles, x-axis first, y-axis second
    aspect : float, optional
        Aspect ratio for the plot

    Returns
    -------
    FacetGrid
        The FacetGrid generated by seaborn
    """

    fcg = sns.lmplot(data=data, x=xcol, y=ycol, col=yidcol, order=degree,
                        col_wrap=col_wrap, aspect=aspect, palette="muted")
    fcg.despine(left=True)
    frmt_xaxislbls(fcg, fmt='{:.2f}', fmt0='{:.2%}',
                 tickscle=1, tickscle0=0.01)
    frmt_yaxislbls(fcg, fmt='{:.1f}', fmt0='{:.1%}',
                 tickscle=1, tickscle0=0.01)
    plt.subplots_adjust(wspace = 0.1)

    return fcg

def defnmodel(data: pd.core.frame.DataFrame, degree: int=1):
    """Defines a polynomial model for data x and y of order degree using
    ordinary least squares regression based

    Parameters
    ----------
    data : panda DataFrame
        Panda data frame containing the data, m rows by 2 columns with the
        first column containing the independent variable (i.e. x) and the second
        column containing the dependent variable (i.e. y)
    degree : int, optional
        The polynomial degree definition e.g. 1 for linear, 2 for a
        quadractic polynomial

    Returns
    -------
    xp : numpy array
        Array of degree columns containing the polynomial features e.g. for a 2
        degree polynomial, features are [1, a, b, a^2, ab, b^2]
    yarray : numpy array
        Array of dependent variable data
    modelresults : sm ols model fit results
        Ordinary least squares regression model produced by statsmodels with
        results
    poly1d_fn : numpy poly1d
        the polynomial definition e.g. x**2 + 2*x + 3
    """

    xcol = 0
    ycol = 1
    polyFeatures = PolynomialFeatures(degree) # Define the polynomial
    # Reshape data from 1 by n to n by 1
    xarray = np.array(data.iloc[:, xcol])
    xarray = xarray[:, np.newaxis]

    # Calculate polynomials for x
    xp = polyFeatures.fit_transform(xarray)

    # Reshape y from 1 by n to n by 1
    yarray = np.array(data.iloc[:, ycol])
    yarray = yarray[:, np.newaxis]

    # Calculate the model and predictions
    modelresults = sm.OLS(yarray, xp).fit()
    coef = modelresults.params.tolist()    # Model coefficients
    coef.reverse()                  # Reverse as poly1d takes in decline order
    poly1d_fn = np.poly1d(coef)     # Create function from coefficients

    return xp, yarray, modelresults, poly1d_fn

def dispmodel(data: pd.core.frame.DataFrame, degree: int=1):
    """Displays summary statistics and regression results for polynomial model
    or order 'degree'

    Parameters
    ----------
    data : panda DataFrame
        Panda data frame containing the data, m rows by 2 columns with the
        first column containing the independent variable (i.e. x) and the second
        column containing the dependent variable (i.e. y)
    degree : int, optional
        The polynomial degree definition e.g. 1 for linear, 2 for a
        quadratic polynomial

    Returns
    -------
    Null
    """

    xcol = 0
    ycol = 1
    xp, yarray, modelresults, poly1d_fn = defnmodel(data, degree)
    print(" ::  FIRST variable (x):")
    print(data.iloc[:, xcol].describe(), '\n')
    print(" ::  SECOND variable (y):")
    print(data.iloc[:, ycol].describe(), '\n')
    print(" :: Pearson Correlation Coefficient:")
    print(data.corr(), '\n\n')
    print(modelresults.summary())

    return
