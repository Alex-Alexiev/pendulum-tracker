import numpy as np
from matplotlib import pyplot as plt 
from scipy import signal
import scipy.optimize as optimize
import math
import errors

def get_relative_x_m(x_px, origin_px, meter_per_px):
    return (x_px-origin_px)*meter_per_px

def get_duplicate_indeces(arr):
    dups = []
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]:
            dups.append(i)
    return dups

def period_length_function(length, k, n, Lo):
    return k*(Lo + length)**n

def analyze(file, length):
    x_px = np.genfromtxt(file + "_x.csv", delimiter=',')
    y_px = np.genfromtxt(file + "_y.csv", delimiter=',')
    r_px = np.genfromtxt(file + "_r.csv", delimiter=',')
    t = np.genfromtxt(file + "_t.csv", delimiter=',')

    avg_radius_px = np.mean(r_px)
    ball_radius_m = 0.06477
    meter_per_px = ball_radius_m/avg_radius_px
    ball_origin_x_px = 1002.8
    #ball_origin_y_px = 786.8
    x_meters = get_relative_x_m(x_px, ball_origin_x_px, meter_per_px)
    length_meters = length/100
    theta = np.asarray([math.asin(x_curr/length) for x_curr in x_meters])
    
    duplicate_indeces = get_duplicate_indeces(theta)
    theta = np.delete(theta, duplicate_indeces)
    time = np.delete(t, duplicate_indeces)
    
    amp_pos_ind = signal.argrelextrema(theta, np.greater, order = 5)
    amp_neg_ind = signal.argrelextrema(theta, np.less, order = 5)

    amp_pos = theta[amp_pos_ind]
    t_amp_pos = time[amp_pos_ind]
    #t_amp_pos = t_amp_pos-t_amp_pos[0] #shift the data so its starts at t=0
    amp_neg = theta[amp_neg_ind]    
    t_amp_neg = time[amp_neg_ind]
    amp_all = np.concatenate((amp_neg[:-1], amp_pos[:-1][::-1]))
    period_neg = np.diff(t_amp_neg, 1)
    period_pos = np.diff(t_amp_pos, 1)    
    periods = np.concatenate((period_neg, period_pos[::-1]))
    p_mean = np.mean(periods)
    std = np.std(periods)
    good_periods = []
    for period in periods:
        if abs(period - p_mean) < std: 
            good_periods.append(period)

    return good_periods

def get_period_length(folder, length):
    periods = analyze(folder + str(length), length)
    return np.mean(periods), np.std(periods)

def analyze_mass_periods():
    periods = []
    period_std = []
    masses = [95, 190, 285, 380, 570, 760]
    for mass in masses:
        period, std = get_period_mass("raw_data/mass/", mass)
        periods.append(period)
        period_std.append(std)
    periods = np.array(periods)
    period_std = np.array(period_std)

    popt, pcov = optimize.curve_fit(power_series, masses, periods)
    A, B, C = popt[0], popt[1], popt[2]
    u_A, u_B, u_C = pcov[0,0]**(0.5), pcov[1,1]**(0.5), pcov[2,2]**(0.5)
    print(A, B, C)
    print(u_A, u_B, u_C)

    start, stop = min(masses), max(masses)
    xs = np.arange(start,stop,(stop-start)/1000) 
    curve = power_series(xs, A, B, C)
    masses = np.asarray(masses)
    residuals = periods - power_series(masses, A, B, C)

    plot_periods_std_mass(masses, periods, period_std)
    plot_data_curve_residuals_mass(masses, periods, xs, curve, residuals)


def analyze_length_periods():
    periods = []
    period_std = []
    lengths = [49, 61, 68, 79, 91, 104, 113, 123, 132, 142, 152, 167]
    for length in lengths:
        period, std = get_period_length("raw_data/length/", length)
        periods.append(period)
        period_std.append(std)
    periods = np.array(periods)
    period_std = np.array(period_std)
    lengths = [49, 61, 68, 79, 91, 101, 113, 123, 132, 142, 152, 167]
    lengths = np.array(lengths)
    lengths = lengths/100


    popt, pcov = optimize.curve_fit(period_length_function, lengths, periods)
    k, n, Lo = popt[0], popt[1], popt[2]
    u_k, u_n, u_Lo = pcov[0,0]**(0.5), pcov[1,1]**(0.5), pcov[2,2]**(0.5)
    print(k, n, Lo)
    print(u_k, u_n, u_Lo)

    start, stop = min(lengths), max(lengths)
    xs = np.arange(start,stop,(stop-start)/1000) 
    curve = period_length_function(xs, k, n, Lo)
    residuals = periods - period_length_function(lengths, k, n, Lo)

    plot_periods_std_length(lengths, periods, period_std)
    plot_norm_log(lengths, periods, xs, curve, residuals)


def plot_data_curve(x, y, x_fit, y_fit, residuals):
    f = plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.errorbar(x, y, yerr=errors.period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)

    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.title("Length VS Period")

    ax = f.add_subplot(1, 2, 2)
    plt.text(0.1, 0.85, "y = 2.0*(0.018 + x)^0.5", transform=ax.transAxes)
    plt.plot(x_fit, y_fit)
    plt.errorbar(x, y, yerr=errors.period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)
    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.title("Power Law Fit Of Periods")

    plt.tight_layout()
    plt.show()

def plot_with_residuals(x, y, x_fit, y_fit, residuals):
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.text(0.1, 0.85, "y = 2.0*(0.018 + x)^0.5", transform=ax.transAxes)
    plt.plot(x_fit, y_fit)
    plt.errorbar(x, y, yerr=errors.period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)
    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.title("Power Law Fit Of Periods")

    zeroliney=[0,0]
    zerolinex=[x[0],x[len(x)-1]]

    plt.subplot(1, 2, 2)
    plt.errorbar(x, residuals,yerr=errors.period,xerr=errors.length,fmt=".")
    plt.plot(zerolinex,zeroliney)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Length (m)")
    plt.ylabel("Residual (s)")
    plt.title("Residuals Of Power Law Fit")

    plt.tight_layout()
    plt.show()

def get_period_mass(folder, mass):
    periods = analyze(folder + str(mass), 127)
    print(periods)
    return np.mean(periods), np.std(periods)

def power_series(x, A, B, C):
    return A + B*x + C*x*x

def plot_periods_std_length(x, y, std):
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.errorbar(x, y, yerr=errors.avg_period, xerr=errors.length, ecolor="red")
    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.legend(("data", ""), loc=4)
    plt.title("Period VS Length")

    upper_error_y=[errors.avg_period,errors.avg_period]
    upper_error_x=[x[0],x[len(x)-1]]

    lower_error_y=[-errors.avg_period,-errors.avg_period]
    lower_error_x=[x[0],x[len(x)-1]]

    plt.subplot(1, 2, 2)
    plt.plot(x, std, "ro", ms=3)
    plt.plot(upper_error_x, upper_error_y)
    plt.plot(lower_error_x, lower_error_y)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Length (m)")
    plt.ylabel("Standard of Deviation (s)")
    plt.legend(("data", "upper period error", "lower period error"), loc=4)
    plt.title("Standard Deviation for Period Data")

    plt.tight_layout()
    plt.show()

def plot_norm_log(x, y, x_fit, y_fit, residuals):
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.text(0.08, 0.85, "y = 2.0*(0.018 + x)^0.5", transform=ax.transAxes)
    plt.plot(x_fit, y_fit)
    plt.errorbar(x, y, yerr=errors.avg_period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)
    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.title("Period VS Length")
    plt.legend(("fit", "data"), loc=4)

    ax = f.add_subplot(1, 2, 2)
    plt.text(0.08, 0.85, "log(y) = 0.5*log(0.018 + x)", transform=ax.transAxes)
    plt.text(0.3, 0.75, "+ log(2.0)", transform=ax.transAxes)
    plt.loglog(x_fit, y_fit)
    plt.errorbar(x, y, yerr=errors.avg_period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)
    plt.xlabel("Length (m)")
    plt.ylabel("Period (s)")
    plt.title("Period VS Length Log-log")
    plt.legend(("fit", "data"), loc=4)
    
    plt.tight_layout()
    plt.show()


def plot_periods_std_mass(x, y, std):
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.errorbar(x, y, yerr=errors.avg_period, xerr=errors.length, ecolor="red")
    plt.xlabel("Mass (g)")
    plt.ylabel("Period (s)")
    plt.ylim((1.9, 2.8))
    plt.legend(("data", ""), loc=4)
    plt.title("Period VS Mass")

    upper_error_y=[errors.avg_period,errors.avg_period]
    upper_error_x=[x[0],x[len(x)-1]]

    lower_error_y=[-errors.avg_period,-errors.avg_period]
    lower_error_x=[x[0],x[len(x)-1]]

    plt.subplot(1, 2, 2)
    plt.plot(x, std, "ro", ms=3)
    plt.plot(upper_error_x, upper_error_y)
    plt.plot(lower_error_x, lower_error_y)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Mass (g)")
    plt.ylabel("Standard of Deviation (s)")
    plt.legend(("data", "upper period error", "lower period error"), loc=4)
    plt.title("Standard Deviation for Period Data")

    plt.tight_layout()
    plt.show()

def plot_data_curve_residuals_mass(x, y, x_fit, y_fit, residuals):
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.text(0.1, 0.85, "y = 2.36 + 6e-5*x - 8e-08*x^2", transform=ax.transAxes)
    plt.plot(x_fit, y_fit)
    plt.errorbar(x, y, yerr=errors.avg_period, xerr=errors.length, ecolor="red", fmt="ro", ms=3)
    plt.xlabel("Mass (g)")
    plt.ylabel("Period (s)")
    plt.ylim((1.9, 2.8))
    plt.legend(("fit", "data"), loc=4)
    plt.title("Power Series Fit Of Periods")

    zeroliney=[0,0]
    zerolinex=[x[0],x[len(x)-1]]

    plt.subplot(1, 2, 2)
    plt.errorbar(x, residuals,yerr=errors.avg_period,xerr=errors.length,fmt=".")
    plt.plot(zerolinex,zeroliney)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Mass (g)")
    plt.ylabel("Residual (s)")
    plt.legend(("zero line", "residuals"), loc=4)
    plt.title("Residuals Of Power Series Fit")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_length_periods()
    analyze_mass_periods()

    