
#Cstats is a tool box for circular statistics and metrics

#numpy is a dependency for this toolbox
import numpy as np

#Find the difference in degrees between directional degrees
def circ_diff_deg(a, b):
    diff = (a - b + 180) % 360 - 180
    return diff

#Find the root mean square error between directional degrees
def circ_rmse(y_true, y_pred):
    diff = circ_diff_deg(y_true, y_pred)
    return np.sqrt(np.mean(diff**2))

#Find the mean of an array of directional degrees
def circ_mean(angles_rad):
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    return np.arctan2(sin_sum, cos_sum)

#Find the correlation between arrays of directional degrees
def circ_correlation(x_deg, y_deg):
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    mean_x = circ_mean(x)
    mean_y = circ_mean(y)
    num = np.sum(np.sin(x-mean_x) * np.sin(y-mean_y))
    den = np.sqrt(
        np.sum(np.sin(x-mean_x)**2) *
        np.sum(np.sin(y-mean_y)**2)
    )
    return num/den

#Find the r2 value between arrays of directional degrees
def circ_r2(x_deg, y_deg):
    rho = circ_correlation(x_deg, y_deg)
    return rho**2

