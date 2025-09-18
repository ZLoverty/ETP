"""
Utility functions for data analysis and processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

def linear(x, a):
    return a * x

def linear_offset(x, a, b):
    return a * x + b

def find_force_transition_velocity(df, velocity_col_name="速度", force_col_name="挤出力", thres=0.01):
    """Fit linear function to the extrusion force data and find the transition from slow increase regime to fast increase regime by identifying the first bad linear fit. 
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the velocity and force data to be analyzed.
    velocity_col_name : str, optional
        Name of the column containing velocity data, by default "速度"
    force_col_name : str, optional
        Name of the column containing force data, by default "挤出力"
    thres : float, optional
        Threshold for the mean squared error to identify bad linear fit, by default 0.01
    
    Returns
    -------
    float
        The velocity at which the transition occurs, or None if not found.
    """

    nPoints_list = range(3, len(df))
    vmax_list = []
    mean_err_list = []
    for nPoints in nPoints_list:
        x = df[velocity_col_name][:nPoints]
        y = df[force_col_name][:nPoints]
        popt, pcov = curve_fit(linear_offset, x, y)
        mean_err = ((y - linear_offset(x, *popt)) ** 2).sum() / nPoints / np.mean(y)**2
        # plt.plot(x, linear_offset(x, *popt), lw=1, ls="--")
        vmax_list.append(x.max())
        mean_err_list.append(mean_err)

    x = np.array(vmax_list)
    y = np.array(mean_err_list)
    if len(x) > 2:
        cs = CubicSpline(x, y, bc_type="natural")
        x_fine = np.linspace(x.min(), x.max(), 100)
        y_fine = cs(x_fine)
        transition_points = x_fine[y_fine > thres]
        if len(transition_points) > 0:
            return transition_points[0]
        else:
            return None
    else:
        raise ValueError("Not enough data points to perform cubic spline interpolation.")
    
def find_steady_state_start(data_series, window_size, threshold):
    """
    Finds the index where a time series reaches a steady state.

    Parameters
    ----------
    data_series : pd.Series
        The time series data.
    window_size : int 
        The number of data points in the sliding window.
    threshold : float 
        The standard deviation threshold to define a steady state.

    Returns
    -------
    int 
        The index of the start of the steady state, or None if not found.
    """
    # reset index, ignore the original index
    data_series = data_series.reset_index(drop=True)
    
    # Calculate the rolling standard deviation
    rolling_std = data_series.rolling(window=window_size).std() / data_series.rolling(window=window_size).mean()

    # Find the first index where the rolling_std is below the threshold
    steady_state_candidates = rolling_std[rolling_std < threshold]

    if not steady_state_candidates.empty:
        # The start of the steady-state REGIME is the first point in that first window
        # So we get the index of the first window that met the criteria
        first_window_end_index = steady_state_candidates.index[0]
        # The actual start of the data regime is the beginning of that window
        steady_state_start_index = max(0, first_window_end_index - window_size + 1)
        return steady_state_start_index
    else:
        return None # No steady state found